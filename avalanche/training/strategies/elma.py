from typing import Optional, Sequence, List, Union

from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer, SGD
import torch.nn as nn

from avalanche.models.pnn import PNN
from avalanche.training.plugins.evaluation import default_logger
from avalanche.training.strategies.base_strategy import BaseStrategy

import copy
import os
import math
import torch
import loralib as lora
import numpy as np
import torch.optim as optim
# from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.training.plugins import StrategyPlugin, EvaluationPlugin, SynapticIntelligencePlugin, GSS_greedyPlugin
from avalanche.models.dynamic_optimizers import reset_optimizer
from collections import defaultdict
from typing import Optional, Sequence, Union, List
from avalanche.benchmarks.scenarios import Experience
from avalanche.evaluation.metrics.eer_metrics import compute_eer
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

class ELMA(BaseStrategy):

    def __init__(self, model: Module, optimizer: Optimizer, criterion,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins: Optional[List[StrategyPlugin]] = None,
                 evaluator: EvaluationPlugin = default_logger, eval_every=-1, split = ""):

        dtype = torch.cuda.FloatTensor  # run on GPU
        self.optimizer = optimizer
        
        self.lora_checkpoints_dir = "/data1/zhangxiaohui/all_dataset/FAD-CL-Benchmark/lora_cpts"
        self.lora_checkpoints_dir = os.path.join(self.lora_checkpoints_dir, split)
        os.makedirs(self.lora_checkpoints_dir, exist_ok=True)
        os.system("rm -f {}/*".format(self.lora_checkpoints_dir))
        
        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)
                

    def initialize_lora_parameters(self, model):
        for name, param in model.named_parameters():
            if 'lora_A' in name:
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            elif "lora_B" in name:
                nn.init.zeros_(param)
                # nn.init.kaiming_uniform_(param, a=math.sqrt(5))
    
    def save_after_exp(self):
        state_dict = self.plugins[1].best_state
        filtered_state_dict = {k: v for k, v in state_dict.items() if "lora_" not in k}
        torch.save(filtered_state_dict, os.path.join(self.lora_checkpoints_dir, "base_model_{}.pt".format(self.current_train_exp_id)))
        # Save current_lora
        lora_A_params = {name: param for name, param in self.model.named_parameters() if 'lora_A' in name}
        lora_A_state_dict = {name: param.clone().detach() for name, param in lora_A_params.items()}
        torch.save(lora_A_state_dict, os.path.join(self.lora_checkpoints_dir, "A_{}.pt".format(self.current_train_exp_id)))
        
        lora_B_params = {name: param for name, param in self.model.named_parameters() if 'lora_B' in name}
        lora_B_state_dict = {name: param.clone().detach() for name, param in lora_B_params.items()}
        torch.save(lora_B_state_dict, os.path.join(self.lora_checkpoints_dir, "B_{}.pt".format(self.current_train_exp_id)))
  
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

        self.model.train()
        self.model.to(self.device)
        
        # Normalize training and eval data.
        if not isinstance(experiences, Sequence):
            experiences = [experiences]
        if eval_streams is None:
            eval_streams = [experiences]
        
        self._before_training(**kwargs)
        
        self._periodic_eval(eval_streams, do_final=False, do_initial=True)

        for self.experience_id, self.experience in enumerate(experiences):
            self.train_exp(self.experience, eval_streams, **kwargs)
        self._after_training(**kwargs)
        
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
        self.experience = experience
        self.current_train_exp_id = experience.current_experience
        self.model.train()

        if eval_streams is None:
            eval_streams = [experience]
        for i, exp in enumerate(eval_streams):
            if not isinstance(exp, Sequence):
                eval_streams[i] = [exp]
                
        self._before_train_dataset_adaptation(**kwargs)
        self.train_dataset_adaptation(**kwargs)
        self._after_train_dataset_adaptation(**kwargs)
        self.make_train_dataloader(**kwargs)

        # Model Adaptation (e.g. freeze/add new units)
        self.model = self.model_adaptation()
        
        # Model Init
        self.initialize_lora_parameters(self.model)
        lora.mark_only_lora_as_trainable(self.model)
            
        # Optimizer Adaptation (e.g. freeze/add new units)
        self.make_optimizer()
        
        self._before_training_exp(**kwargs)
        
        do_final = True
        if self.eval_every > 0 and \
                (self.train_epochs - 1) % self.eval_every == 0:
            do_final = False

        for _ in range(self.train_epochs):
            self._before_training_epoch(**kwargs)
            
            if self._stop_training:  # Early stopping
                self._stop_training = False
                break

            self.training_epoch(**kwargs)
            
            self._after_training_epoch(**kwargs)
            self._periodic_eval(eval_streams, do_final=False)
        
        self.save_after_exp()
        # Final evaluation
        self._periodic_eval(eval_streams, do_final=do_final)
        self._after_training_exp(**kwargs)
          
    def load_for_eval(self, lora_id):
        print("load chpts from lora ID {} ...".format(lora_id))
        # Load the pretrained checkpoint first
        self.model.load_state_dict(torch.load(os.path.join(self.lora_checkpoints_dir, "base_model_{}.pt".format(lora_id))), strict=False)
        # Then load the LoRA checkpoint
        self.model.load_state_dict(torch.load(os.path.join(self.lora_checkpoints_dir, "A_{}.pt".format(lora_id))), strict=False)
        self.model.load_state_dict(torch.load(os.path.join(self.lora_checkpoints_dir, "B_{}.pt".format(lora_id))), strict=False)
 
    def training_epoch(self, lora_lr_scale=100, **kwargs):
        """ Training epoch with scaled learning rate for LoRA parameters.
        
        :param lora_lr_scale: The scaling factor for the learning rate of LoRA parameters
        :param kwargs:
        :return:
        """
        # Initialize the optimizer with a higher learning rate for LoRA parameters
        # Assuming you have separate optimizers for lora_A and lora_B

        for self.i_batch, self.mbatch in enumerate(self.dataloader):
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            # Reset gradients before each batch
            self.optimizer.zero_grad()
            
            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            self._after_forward(**kwargs)

            # Loss & Backward
            self.losses = self.criterion()
            self.loss = self.losses.mean()
            
            self._before_backward(**kwargs)
            
            self.loss.backward()
               
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer.step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

    def ensemble_predictions(self, model_predictions):
        n = len(model_predictions)
        if n == 1:
            return model_predictions[0]
        
        num_samples = model_predictions[0].shape[0]
        # model_predictions = np.array(model_predictions)
        weighted_predictions = np.zeros(num_samples)

        variances = np.array([np.var(pred) for pred in model_predictions])

        weights = variances / np.sum(variances)

        for i in range(n):
            weighted_predictions += model_predictions[i] * weights[i]
        
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

        self._before_eval(**kwargs)
        for self.experience in exp_list:
            # Data Adaptation
            self._before_eval_dataset_adaptation(**kwargs)
            self.eval_dataset_adaptation(**kwargs)
            self._after_eval_dataset_adaptation(**kwargs)
            self.make_eval_dataloader(**kwargs)

            for lora_id in range(self.current_train_exp_id + 1):
                # Model Adaptation (e.g. freeze/add new units)
                self.model = self.model_adaptation()

                self._before_eval_exp(**kwargs)
                """
                Load each lora to infer
                """
                self.load_for_eval(lora_id)
                self.eval_epoch(**kwargs)
                self._after_eval_exp(**kwargs)
            
                scores = torch.cat(self.score_loader, 0).data.cpu().numpy()
                self.exp_score_loader.append(scores)
                labels = torch.cat(self.target_loader, 0).data.cpu().numpy()
                eer = compute_eer(scores[labels == 0], scores[labels == 1])[0]
                other_eer = compute_eer(-scores[labels == 0], -scores[labels == 1])[0]
                eer_cm = min(eer, other_eer)

                print('===> Exp: {} -- lora_id: {} -- EER_CM: {}\n'.format(self.experience.current_experience, lora_id, eer_cm))
            """
            MoE Essemble
            """
            print("Predication ensembling ...")
            scores = self.ensemble_predictions(self.exp_score_loader)
            labels = torch.cat(self.target_loader, 0).data.cpu().numpy()
            
            eer = compute_eer(scores[labels == 0], scores[labels == 1])[0]
            other_eer = compute_eer(-scores[labels == 0], -scores[labels == 1])[0]
            eer_cm = min(eer, other_eer)

            print('===> Exp: {} EER_CM: {}\n'.format(self.experience.current_experience, eer_cm))
            
            self.exp_score_loader = []
            
            # Record it
            self.eer_record[self.clock.train_exp_counter].append(eer_cm)
        
        self._after_eval(**kwargs)
        
        res = self.evaluator.get_last_metrics()

        return res
    
    # def criterion(self, mb_output = None, mb_y = None):
    #     """ Loss function. """
    #     mb_output = mb_output if mb_output is not None else self.mb_output
    #     mb_y = mb_y if mb_y is not None else self.mb_y
    #     if isinstance(mb_output, tuple):
    #         mb_output = mb_output[0]
    #     return self._criterion(mb_output, mb_y)

__all__ = ['ELMA']