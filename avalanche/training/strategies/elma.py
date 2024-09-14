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
        # param_group = optimizer.param_groups[0]

        # Get Lr and Wd
        # self.lora_learning_rate = param_group['lr'] * 100
        # self.lora_weight_decay = param_group['weight_decay']

        # self.optimizer_A =  optim.Adam((param for name, param in model.named_parameters() if 'lora_A' in name), 
        #                     lr = self.lora_learning_rate, weight_decay = self.lora_weight_decay)
        # self.optimizer_B =  optim.Adam((param for name, param in model.named_parameters() if 'lora_B' in name), 
        #                     lr = self.lora_learning_rate, weight_decay = self.lora_weight_decay)
        
        self.lora_checkpoints_dir = "/data1/zhangxiaohui/all_dataset/FAD-CL-Benchmark/lora_cpts"
        self.lora_checkpoints_dir = os.path.join(self.lora_checkpoints_dir, split)
        os.makedirs(self.lora_checkpoints_dir, exist_ok=True)
        os.system("rm -f {}/*".format(self.lora_checkpoints_dir))
        
        # self.scheduler_A = None
        # self.scheduler_B = None
        
        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)
    
    def Init_use_best_A(self):
        """
        Use best lora_A to initialize the current lora_A
        """
        print("==========> Choosing best LoRA_A.... ==========>")
        if self.current_train_exp_id == 1:
            self.model.load_state_dict(torch.load(os.path.join(self.lora_checkpoints_dir, "A_0.pt")), strict=False)
            print("++++++++++ Picked LoRA_A_0 ++++++++++")
        else:
            """
            Forward each sample in current experience training set and calculate loss
            Use the min loss as the metric to choose the best previous lora_A
            """
            loss_list = [0] * self.current_train_exp_id
            for lora_id in range(self.current_train_exp_id):
                self.load_for_eval(lora_id)
                print("Choosing checking lora {}".format(lora_id))
                for self.i_batch, self.mbatch in enumerate(self.dataloader):
                    self._unpack_minibatch()
                    # Forward
                    self.mb_output = self.forward()
                    # Loss
                    loss_list[lora_id] += self.criterion()
            best_lora_id = loss_list.index(min(loss_list))
            print("++++++++++ Picked LoRA_A_{} ++++++++++".format(best_lora_id))
            self.initialize_lora_parameters(self.model) # Reset LoRA_B
            self.model.load_state_dict(torch.load(os.path.join(self.lora_checkpoints_dir, "A_{}.pt".format(best_lora_id))), strict=False)                  

    def initialize_lora_parameters(self, model):
        for name, param in model.named_parameters():
            if 'lora_A' in name:
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            elif "lora_B" in name:
                nn.init.zeros_(param)
                # nn.init.kaiming_uniform_(param, a=math.sqrt(5))
    
    def save_after_exp(self):
        # if self.current_train_exp_id == 0:
            # torch.save(self.model.state_dict(), os.path.join(self.lora_checkpoints_dir, "base_model.pt"))
        # state_dict = self.model.state_dict()
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
        
    def mymake_optimizer(self):
        """
        Initialize optimizer when first experience;
        and
        Initialize only optimizer_A and ..._B in other experiences (only train lora_A and lora_B on experience 1, 2,...)
        """
        if self.current_train_exp_id == 0:
            reset_optimizer(self.optimizer, self.model)
        else:
            # Initialize the state and parameters for optimizer_A
            self.optimizer_A.state = defaultdict(dict)
            params_A = [param for name, param in self.model.named_parameters() if 'lora_A' in name]
            for param in params_A:
                param.requires_grad = True
            self.optimizer_A.param_groups[0]['params'] = params_A
            self.optimizer_A.param_groups[0]['lr'] = self.lora_learning_rate
            self.optimizer_A.param_groups[0]['weight_decay'] = self.lora_weight_decay

            # Initialize the state and parameters for optimizer_B
            self.optimizer_B.state = defaultdict(dict)
            params_B = [param for name, param in self.model.named_parameters() if 'lora_B' in name]
            for param in params_B:
                param.requires_grad = True
            self.optimizer_B.param_groups[0]['params'] = params_B
            self.optimizer_B.param_groups[0]['lr'] = self.lora_learning_rate
            self.optimizer_B.param_groups[0]['weight_decay'] = self.lora_weight_decay
            # Initialize learning rate schedulers for optimizer_A and optimizer_B
            self.scheduler_A = StepLR(self.optimizer_A, step_size = 10, gamma = 1)
            self.scheduler_B = StepLR(self.optimizer_B, step_size = 10, gamma = 1)
    
    def step_schedulers(self):
        if self.scheduler_A:
            self.scheduler_A.step()
        if self.scheduler_B:
            self.scheduler_B.step()
    
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
            # import pdb; pdb.set_trace()
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
        # if self.current_train_exp_id != 0:
        #     self.Init_use_best_A()
            
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
            # Reduce the lr on lora_A/B
            # self.step_schedulers()
            
            self._after_training_epoch(**kwargs)
            self._periodic_eval(eval_streams, do_final=False)
        
        self.save_after_exp()
        # Final evaluation
        self._periodic_eval(eval_streams, do_final=do_final)
        self._after_training_exp(**kwargs)

    def mytraining_epoch_v1(self, **kwargs):
        """ Training epoch.
        
        :param kwargs:
        :return:
        """
        for self.i_batch, self.mbatch in enumerate(self.dataloader):
            # self.dataloader._dl.datasets[0]._indices
            if self._stop_training:
                break
            
            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)
            if self.current_train_exp_id == 0:
                """
                Train model and lora_A/B_1
                """
                # import pdb; pdb.set_trace()
                # print("===========================> Train base and lora ...")
                self.optimizer.zero_grad()
                self.loss = 0

                # Forward
                self._before_forward(**kwargs)
                self.mb_output = self.forward()
                self._after_forward(**kwargs)

                # Loss & Backward
                self.loss += self.criterion()

                self._before_backward(**kwargs)
                self.loss.backward()
                self._after_backward(**kwargs)

                # Optimization step
                self._before_update(**kwargs)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self._after_update(**kwargs)
            else:
                """
                Only train lora_A/B_n
                """
                # print("===========================> Train only lora ...")
                # print(self.mb_task_id)
                input_f = torch.stack([self.mb_x[i].to(self.device) for i in range(len(self.mb_x)) if self.mb_y[i] == 1])
                input_r = torch.stack([self.mb_x[i].to(self.device) for i in range(len(self.mb_x)) if self.mb_y[i] == 0])
                label_f = torch.tensor([self.mb_y[i] for i in range(len(self.mb_y)) if self.mb_y[i] == 1]).to(self.device)
                label_r = torch.tensor([self.mb_y[i] for i in range(len(self.mb_y)) if self.mb_y[i] == 0]).to(self.device)
                # import pdb; pdb.set_trace()
                self.optimizer_A.zero_grad()
                self.optimizer_B.zero_grad()
                self.loss = 0
                mb_out_list = []
                # Forward Fake
                self._before_forward(**kwargs)
                self.mb_output = self.model(input_f)
                mb_out_list.append(self.mb_output)
                self._after_forward(**kwargs)
                
                # Loss & Backward
                loss = self.criterion(mb_y = label_f)
                
                self._before_backward(**kwargs)
                loss.backward()
                # print("Lora_A.requires_grad: {} grad==0: {}".format(self.model.module.classifier.lora_A.requires_grad, torch.all(self.model.module.classifier.lora_A.grad==0)))
                # if not torch.all(self.model.module.classifier.lora_A.grad==0):
                    # import pdb; pdb.set_trace()
                self._after_backward(**kwargs)
                
                # Optimize Lora B
                self._before_update(**kwargs)
                # for n, w in self.model.named_parameters():
                #     # if "lora_B" in n:
                #     w.grad.data = w.grad.data * 100
                self.optimizer_B.step()
                self._after_update(**kwargs)
                
                self.optimizer_A.zero_grad()
                self.optimizer_B.zero_grad()
                self.loss = 0
                
                # Forward Real
                self._before_forward(**kwargs)
                self.mb_output = self.model(input_r)
                mb_out_list.append(self.mb_output)
                self._after_forward(**kwargs)
                
                # Loss & Backward
                self.loss += self.criterion(mb_y = label_r)
                
                self._before_backward(**kwargs)
                self.loss.backward()
                # print("Lora_A.requires_grad: {} grad==0: {}".format(self.model.module.classifier.lora_A.requires_grad, torch.all(self.model.module.classifier.lora_A.grad==0)))
                self._after_backward(**kwargs)
                
                # Optimize Lora A
                self._before_update(**kwargs)
                # for n, w in self.model.named_parameters():
                #     w.grad.data = w.grad.data * 100
                self.optimizer_A.step() 
                self._after_update(**kwargs)
                
                self.mb_output = torch.cat(mb_out_list, dim = 0)
            self._after_training_iteration(**kwargs)         

    def mytraining_epoch_v2(self, **kwargs):
        """ Training epoch.
        
        :param kwargs:
        :return:
        """
        for self.i_batch, self.mbatch in enumerate(self.dataloader):
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = 0

            # Forward
            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            self._after_forward(**kwargs)

            # Loss & Backward
            self.loss += self.criterion()
            self._before_backward(**kwargs)
            self.loss.backward()

            # 参数选择性更新逻辑
            if self.current_train_exp_id != 0:
                target_param_name = "lora_A" if self.mb_y == 0 else "lora_B"
                for name, param in self.model.named_parameters():
                    if target_param_name not in name:
                        param.grad = None  # 清除不需要更新的参数的梯度

            self.optimizer.step()  # 统一在此处更新参数

            self._after_backward(**kwargs)
            self._before_update(**kwargs)
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)
            
    def load_for_eval(self, lora_id):
        print("load chpts from lora ID {} ...".format(lora_id))
        # Load the pretrained checkpoint first
        self.model.load_state_dict(torch.load(os.path.join(self.lora_checkpoints_dir, "base_model_{}.pt".format(lora_id))), strict=False)
        # Then load the LoRA checkpoint
        self.model.load_state_dict(torch.load(os.path.join(self.lora_checkpoints_dir, "A_{}.pt".format(lora_id))), strict=False)
        self.model.load_state_dict(torch.load(os.path.join(self.lora_checkpoints_dir, "B_{}.pt".format(lora_id))), strict=False)

    def mytraining_epoch_v3(self, **kwargs):
        """ Training epoch.
        
        :param kwargs:
        :return:
        """
        for self.i_batch, self.mbatch in enumerate(self.dataloader):
            if self._stop_training:
                break
            
            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            # Reset gradients before each batch
            self.optimizer.zero_grad()

            if self.current_train_exp_id == 0:
                """
                Train model and lora_A/B_1
                """
                self.loss = 0

                # Forward
                self._before_forward(**kwargs)
                self.mb_output = self.forward()
                self._after_forward(**kwargs)

                # Loss & Backward
                self.loss += self.criterion()

                self._before_backward(**kwargs)
                self.loss.backward()
                self._after_backward(**kwargs)

                # Optimization step
                self._before_update(**kwargs)
                self.optimizer.step()
                self._after_update(**kwargs)

            else:
                """
                Only train lora_A/B_n with conditional freezing of lora_A based on label
                """
                input_f = torch.stack([self.mb_x[i].to(self.device) for i in range(len(self.mb_x)) if self.mb_y[i] == 1])
                input_r = torch.stack([self.mb_x[i].to(self.device) for i in range(len(self.mb_x)) if self.mb_y[i] == 0])
                label_f = torch.tensor([self.mb_y[i] for i in range(len(self.mb_y)) if self.mb_y[i] == 1]).to(self.device)
                label_r = torch.tensor([self.mb_y[i] for i in range(len(self.mb_y)) if self.mb_y[i] == 0]).to(self.device)
                mb_out_list = []

                # Handle Fake samples (label=1): Train Lora B
                # Ensure lora_A's requires_grad is True (unfrozen)
                for name, param in self.model.named_parameters():
                    if 'lora_A' in name:
                        param.requires_grad = False

                # Forward Fake
                self._before_forward(**kwargs)
                self.mb_output = self.model(input_f)
                mb_out_list.append(self.mb_output)
                self._after_forward(**kwargs)

                # Compute Loss for Fake samples
                self.loss = self.criterion(mb_y=label_f)

                # Backward and optimize
                self._before_backward(**kwargs)
                self.loss.backward()
                self._after_backward(**kwargs)
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Handle Real samples (label=0): Train Lora A
                # Freeze lora_A parameters
                for name, param in self.model.named_parameters():
                    if 'lora_A' in name:
                        param.requires_grad = True

                # Forward Real
                self._before_forward(**kwargs)
                self.mb_output = self.model(input_r)
                mb_out_list.append(self.mb_output)
                self._after_forward(**kwargs)

                # Compute Loss for Real samples
                self.loss = self.criterion(mb_y=label_r)

                # Backward and optimize
                self._before_backward(**kwargs)
                self.loss.backward()
                self._after_backward(**kwargs)
                self.optimizer.step()

                # Combine outputs
                self.mb_output = torch.cat(mb_out_list, dim=0)

            self._after_training_iteration(**kwargs)
    
    def mytraining_epoch_v4(self, lora_lr_scale=100, **kwargs):
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

            if self.current_train_exp_id != -1:
                """
                Train model and lora_A/B_1
                """
                self.loss = 0
                # import pdb; pdb.set_trace()
                # Forward
                self._before_forward(**kwargs)
                self.mb_output = self.forward()
                self._after_forward(**kwargs)

                # Loss & Backward
                self.loss += self.criterion()

                self._before_backward(**kwargs)
                self.loss.backward()
                self._after_backward(**kwargs)

                # Optimization step
                self._before_update(**kwargs)
                self.optimizer.step()
                self._after_update(**kwargs)

            else:
                """
                Only train lora_A/B_n with learning rate scaling for lora parameters
                """
                # import pdb; pdb.set_trace()
                input_f = torch.stack([self.mb_x[i].to(self.device) for i in range(len(self.mb_x)) if self.mb_y[i] == 1])
                input_r = torch.stack([self.mb_x[i].to(self.device) for i in range(len(self.mb_x)) if self.mb_y[i] == 0])
                label_f = torch.tensor([self.mb_y[i] for i in range(len(self.mb_y)) if self.mb_y[i] == 1]).to(self.device)
                label_r = torch.tensor([self.mb_y[i] for i in range(len(self.mb_y)) if self.mb_y[i] == 0]).to(self.device)
                mb_out_list = []

                # Handle Fake samples (label=1): Train Lora B
                # Freeze lora_A parameters
                for name, param in self.model.named_parameters():
                    if 'lora_A' in name:
                        param.requires_grad = True # False

                # Forward Fake
                self._before_forward(**kwargs)
                self.mb_output = self.model(input_f)
                mb_out_list.append(self.mb_output)
                self._after_forward(**kwargs)

                # Compute Loss for Fake samples
                self.loss = self.criterion(mb_y=label_f)

                # Backward and optimize only LoRA B parameters
                self._before_backward(**kwargs)
                self.loss.backward()

                # Scale the gradients for lora_B by the specified learning rate factor
                # for name, param in self.model.named_parameters():
                #     if 'lora_B' in name and param.requires_grad:
                #         param.grad.data.mul_(lora_lr_scale)

                self._after_backward(**kwargs)
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Handle Real samples (label=0): Train Lora A
                # Freeze lora_B parameters
                for name, param in self.model.named_parameters():
                    if 'lora_A' in name:
                        param.requires_grad = True

                # Forward Real
                self._before_forward(**kwargs)
                self.mb_output = self.model(input_r)
                mb_out_list.append(self.mb_output)
                self._after_forward(**kwargs)

                # Compute Loss for Real samples
                self.loss = self.criterion(mb_y=label_r)

                # Backward and optimize only LoRA A parameters
                self._before_backward(**kwargs)
                self.loss.backward()

                # Scale the gradients for lora_A/B by the specified learning rate factor
                # for name, param in self.model.named_parameters():
                #     if 'lora_' in name and param.requires_grad:
                #         param.grad.data.mul_(lora_lr_scale)

                self._after_backward(**kwargs)
                self.optimizer.step()

                # Combine outputs
                self.mb_output = torch.cat(mb_out_list, dim=0)

            self._after_training_iteration(**kwargs)
    
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
            
            if self.current_train_exp_id != 100:
                """
                Train lora_A/B
                """
                # Compute gradients
                self.loss.backward()
            else:
                for i in range(len(self.mb_x)):
                    # self.optimizer.zero_grad()
                    label_sample = self.mb_y[i]
                    loss_sample = self.losses[i]/len(self.mb_x)

                    # Freeze lora_A parameters if label == 1 (fake sample)
                    for name, param in self.model.named_parameters():
                        if 'lora_A' in name:
                            param.requires_grad = False if label_sample == 1 else True

                    # Backward pass to compute gradients for this sample
                    loss_sample.backward(retain_graph=True)  # Compute gradient for this sample
               
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer.step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)
    
    def mytraining_epoch_v6(self, lora_lr_scale=100, **kwargs):
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

            # Before forward, adjust requires_grad based on the label
            assert len(self.mb_x) == 1  # Assuming batch_size=1
            label_sample = self.mb_y[0]
            for name, param in self.model.named_parameters():
                if 'lora_A' in name:
                    param.requires_grad = False if label_sample == 1 else True

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
           
    def calculate_entropy(self, probabilities):
        epsilon = 1e-10 
        return -np.sum(probabilities * np.log(probabilities + epsilon) + (1 - probabilities) * np.log(1 - probabilities + epsilon))

    def calculate_weights(self, entropies):
        weights = 1 / (entropies + 1e-10)
        normalized_weights = F.softmax(torch.tensor(weights), dim=0).numpy()
        return normalized_weights

    # def ensemble_predictions(self, model_predictions):
    #     n = len(model_predictions)
    #     if n == 1:
    #         return model_predictions[0]
    #     num_samples = model_predictions[0].shape[0]
    #     # model_predictions = np.array([t.numpy() for t in model_predictions])
    #     model_predictions = np.array(model_predictions)
    #     weighted_predictions = np.zeros(num_samples)
    #     for i in range(num_samples):           
    #         sample_predictions = model_predictions[:, i]            
    #         entropies = np.array([self.calculate_entropy(pred) for pred in sample_predictions])
    #         # import pdb; pdb.set_trace()
    #         weights = self.calculate_weights(entropies)
    #         weighted_predictions[i] = np.sum(sample_predictions * weights)
    #     return weighted_predictions 

    def ensemble_predictions(self, model_predictions):
        n = len(model_predictions)
        if n == 1:
            return model_predictions[0]
        
        # import pdb; pdb.set_trace()
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
                # lora_id = self.current_train_exp_id
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
            # if self.current_train_exp_id == 1:
            #     import pdb; pdb.set_trace()
            print("Predication ensembling ...")
            scores = self.ensemble_predictions(self.exp_score_loader)
            # scores = self.exp_score_loader[-1]
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