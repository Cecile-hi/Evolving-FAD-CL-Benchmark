import copy
import os
import torch
import loralib as lora
import numpy as np
import torch.optim as optim
from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.models.dynamic_optimizers import reset_optimizer
from collections import defaultdict
from typing import Optional, Sequence, Union, List
from avalanche.benchmarks.scenarios import Experience
from avalanche.evaluation.metrics.eer_metrics import compute_eer
import torch.nn.functional as F

# from avalanche.training import BaseStrategy


class ELMAPlugin(StrategyPlugin):
# class ELMAPlugin(BaseStrategy):
    """
    A Learning without Forgetting plugin.
    LwF uses distillation to regularize the current loss with soft targets
    taken from a previous version of the model.
    This plugin does not use task identities.
    When used with multi-headed models, all heads are distilled.
    """


    def __init__(self, model, optimizer):
        """
        :param alpha: distillation hyperparameter. It can be either a float
                number or a list containing alpha for each experience.
        :param temperature: softmax temperature for distillation
        """

        super().__init__()

        dtype = torch.cuda.FloatTensor  # run on GPU
        self.optimizer = optimizer
        param_group = optimizer.param_groups[0]

        # Get Lr and Wd
        learning_rate = param_group['lr']
        weight_decay = param_group['weight_decay']

        self.optimizer_A =  optim.Adam((param for name, param in model.named_parameters() if 'lora_A' in name), 
                            lr=learning_rate, weight_decay=weight_decay)
        self.optimizer_B =  optim.Adam((param for name, param in model.named_parameters() if 'lora_B' in name), 
                            lr=learning_rate, weight_decay=weight_decay)
        
        self.lora_checkpoints_dir = "/data1/zhangxiaohui/all_dataset/FAD-CL-Benchmark/lora_cpts"
        """ In Avalanche, targets of different experiences are not ordered. 
        As a result, some units may be allocated even though their 
        corresponding class has never been seen by the model.
        Knowledge distillation uses only units corresponding to old classes. 
        """
    
    def Init_use_best_A(self):
        """
        Use best lora_A to initialize the current lora_A
        """
        if self.current_train_exp_id == 1:
            self.model.load_state_dict(torch.load(os.path.join(self.lora_checkpoints_dir, "A_0.pt")), strict=False)
        else:
            """
            Forward each sample in current experience training set and calculate loss
            Use the min loss as the metric to choose the best previous lora_A
            """
            loss_list = [0] * self.current_train_exp_id
            for self.i_batch, self.mbatch in enumerate(self.dataloader):
                for lora_id in range(self.current_train_exp_id):
                    self.load_for_eval(lora_id)
                    # Forward
                    self.mb_output = self.forward()
                    # Loss
                    loss_list[lora_id] += self.criterion()
            best_lora_id = loss_list.index(min(loss_list))
            self.model.load_state_dict(torch.load(os.path.join(self.lora_checkpoints_dir, "A_{}.pt".format(best_lora_id))), strict=False)
    
    def reset_lora(self):
        for name, param in self.model.named_parameters():
            if 'lora_' in name:
                param.data.zero_()                      
    
    def save_after_exp(self):
        if self.current_train_exp_id == 0:
            # torch.save(self.model.state_dict(), os.path.join(self.lora_checkpoints_dir, "base_model.pt"))
            state_dict = self.model.state_dict()
            filtered_state_dict = {k: v for k, v in state_dict.items() if "lora_" not in k}
            torch.save(filtered_state_dict, os.path.join(self.lora_checkpoints_dir, "base_model.pt"))
        # Save current_lora
        lora_A_params = {name: param for name, param in self.model.named_parameters() if 'lora_A' in name}
        lora_A_state_dict = {name: param.clone().detach() for name, param in lora_A_params.items()}
        torch.save(lora_A_state_dict, os.path.join(self.lora_checkpoints_dir, "A_{}.pt".format(self.current_train_exp_id)))
        
        lora_B_params = {name: param for name, param in self.model.named_parameters() if 'lora_B' in name}
        lora_B_state_dict = {name: param.clone().detach() for name, param in lora_B_params.items()}
        torch.save(lora_B_state_dict, os.path.join(self.lora_checkpoints_dir, "B_{}.pt".format(self.current_train_exp_id)))
        
    def make_optimizer(self):
        """
        Initialize optimizer when first experience;
        and
        Initialize only optimizer_A and ..._B in other experiences (only train lora_A and lora_B on experience 1, 2,...)
        """
        if self.current_train_exp_id == 0:
            reset_optimizer(self.optimizer, self.model)
        else:
            self.optimizer_A.state = defaultdict(dict)
            self.optimizer_A.param_groups[0]['params'] = list(param for name, param in self.model.named_parameters() if 'lora_A' in name)
            self.optimizer_B.state = defaultdict(dict)
            self.optimizer_B.param_groups[0]['params'] = list(param for name, param in self.model.named_parameters() if 'lora_B' in name)
    
    def train(self, experiences: Union[Experience, Sequence[Experience]],
              eval_streams: Optional[Sequence[Union[Experience, Sequence[Experience]]]] = None, **kwargs):
        """ Training loop. if experiences is a single element trains on it.
        If it is a sequence, trains the model on each experience in order.
        This is different from joint training on the entire stream.
        It returns a dictionary with last recorded value for each metric.

        :param experiences: single Experience or sequence.
        :param eval_streams: list of streams for evaluation.
            If None: use training experiences for evaluation.
            Use [] if you do not want to evaluate during training.

        :return: dictionary containing last recorded value for
            each metric name.
        """
        self.is_training = True
        self._stop_training = False
        import pdb
        pdb.set_trace()
        self.model.train()
        self.model.to(self.device)
        
        # Normalize training and eval data.
        if not isinstance(experiences, Sequence):
            experiences = [experiences]
        if eval_streams is None:
            eval_streams = [experiences]

        for self.experience_id, self.experience in enumerate(experiences):
            self.train_exp(self.experience, eval_streams, **kwargs)

        res = self.evaluator.get_last_metrics()
        return res
    
    def train_exp(self, experience: Experience, eval_streams=None, **kwargs):
        """ Training loop over a single Experience object.

        :param experience: CL experience information.
        :param eval_streams: list of streams for evaluation.
            If None: use the training experience for evaluation.
            Use [] if you do not want to evaluate during training.
        :param kwargs: custom arguments.
        """
        # self.experience = experience
        self.current_train_exp_id = experience.current_experience
        
        # if self.current_train_exp_id != 0:
        #     lora.mark_only_lora_as_trainable(self.model)

        self.model.train()

        if eval_streams is None:
            eval_streams = [experience]
        for i, exp in enumerate(eval_streams):
            if not isinstance(exp, Sequence):
                eval_streams[i] = [exp]
                
        self.train_dataset_adaptation(**kwargs)

        self.make_train_dataloader(**kwargs)

        # Model Adaptation (e.g. freeze/add new units)
        self.model = self.model_adaptation()
        
        # Model Init
        if self.current_train_exp_id != 0:
            self.reset_lora()
            self.Init_use_best_A()

        # Optimizer Adaptation (e.g. freeze/add new units)
        self.make_optimizer()
        
        do_final = True
        if self.eval_every > 0 and \
                (self.train_epochs - 1) % self.eval_every == 0:
            do_final = False

        for _ in range(self.train_epochs):
            if self._stop_training:  # Early stopping
                self._stop_training = False
                break

            self.training_epoch(**kwargs)
            self._periodic_eval(eval_streams, do_final=False)
        
        self.save_after_exp()
        # Final evaluation
        self._periodic_eval(eval_streams, do_final=do_final)

    def training_epoch(self, **kwargs):
        """ Training epoch.
        
        :param kwargs:
        :return:
        """ 
        for self.i_batch, self.mbatch in enumerate(self.dataloader):
            # self.dataloader._dl.datasets[0]._indices
            if self._stop_training:
                break
            if self.current_train_exp_id == 0:
                """
                Train model and lora_A/B_1
                """
                self.optimizer.zero_grad()
                self.loss = 0

                # Forward
                self.mb_output = self.forward()

                # Loss & Backward
                self.loss += self.criterion()

                self.loss.backward()

                # Optimization step
                self.optimizer.step()
            else:
                """
                Only train lora_A/B_n
                """
                input_f = [self.mbatch[i].to(self.device) for i in range(len(self.mbatch)) if self.mb_y[i] == 0]
                input_f = torch.tensor(input_f).to(self.device)
                input_r = [self.mbatch[i].to(self.device) for i in range(len(self.mbatch)) if self.mb_y[i] == 1]
                input_r = torch.tensor(input_r).to(self.device)
                
                self.optimizer_A.zero_grad()
                self.optimizer_B.zero_grad()
                self.loss = 0
                
                self.mb_output = self.model(input_f)
                self.loss += self.criterion()
                self.loss.backward()
                self.optimizer_B.step()
                
                self.optimizer_A.zero_grad()
                self.optimizer_B.zero_grad()
                self.loss = 0
                
                self.mb_output = self.model(input_r)
                self.loss += self.criterion()
                self.loss.backward()
                self.optimizer_A.step()             

    def load_for_eval(self, lora_id):
        # Load the pretrained checkpoint first
        self.model.load_state_dict(torch.load(os.path.join(self.lora_checkpoints_dir, "base_model.pt")), strict=False)
        # Then load the LoRA checkpoint
        self.model.load_state_dict(torch.load(os.path.join(self.lora_checkpoints_dir, "A_{}.pt".format(lora_id))), strict=False)
        self.model.load_state_dict(torch.load(os.path.join(self.lora_checkpoints_dir, "B_{}.pt".format(lora_id))), strict=False)
                
    def calculate_entropy(self, probabilities):
        epsilon = 1e-10 
        return -np.sum(probabilities * np.log(probabilities + epsilon) + (1 - probabilities) * np.log(1 - probabilities + epsilon))

    def calculate_weights(self, entropies):
        weights = 1 / (entropies + 1e-10)  # 避免除以0
        normalized_weights = F.softmax(torch.tensor(weights), dim=0).numpy()
        return normalized_weights

    def ensemble_predictions(self, model_predictions):
        n = len(model_predictions)
        num_samples = model_predictions[0].shape[0]
        
        model_predictions = np.array(model_predictions)
        weighted_predictions = np.zeros(num_samples)

        for i in range(num_samples):
            sample_predictions = model_predictions[:, i]            
            entropies = np.array([self.calculate_entropy(pred) for pred in sample_predictions])
            weights = self.calculate_weights(entropies)
            weighted_predictions[i] = np.sum(sample_predictions * weights)
        return weighted_predictions
        
    
    @torch.no_grad()
    def eval(self, exp_list: Union[Experience, Sequence[Experience]], **kwargs):
        """
        Evaluate the current model on a series of experiences and
        returns the last recorded value for each metric.

        :param exp_list: CL experience information.
        :param kwargs: custom arguments.

        :return: dictionary containing last recorded value for
            each metric name
        """
        self.is_training = False
        self.model.eval()
        
        self.exp_score_loader = []

        if not isinstance(exp_list, Sequence):
            exp_list = [exp_list]
        self.current_eval_stream = exp_list

        for self.experience in exp_list:
            # Data Adaptation
            self.eval_dataset_adaptation(**kwargs)
            
            self.make_eval_dataloader(**kwargs)

            for lora_id in range(self.current_train_exp_id + 1):
                """
                Load each lora to infer
                """
                self.load_for_eval(lora_id)
                # Model Adaptation (e.g. freeze/add new units)
                self.model = self.model_adaptation()

                self.eval_epoch(**kwargs)
            
                scores = torch.cat(self.score_loader, 0).data.cpu().numpy()
                self.exp_score_loader.append(scores)
            """
            MoE Essemble
            """
            scores = self.ensemble_predictions(self.exp_score_loader)
            
            labels = torch.cat(self.target_loader, 0).data.cpu().numpy()
            
            eer = compute_eer(scores[labels == 0], scores[labels == 1])[0]
            other_eer = compute_eer(-scores[labels == 0], -scores[labels == 1])[0]
            eer_cm = min(eer, other_eer)

            print('===> Exp: {} EER_CM: {}\n'.format(self.experience.current_experience, eer_cm))
            
            # Record it
            self.eer_record[self.clock.train_exp_counter].append(eer_cm)
        
        self._after_eval(**kwargs)
        
        res = self.evaluator.get_last_metrics()

        return res
