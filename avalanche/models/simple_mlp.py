################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 1-05-2020                                                              #
# Author(s): Vincenzo Lomonaco, Antonio Carta                                  #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

import torch.nn as nn
import loralib as lora
from avalanche.models.dynamic_modules import MultiTaskModule, \
    MultiHeadClassifier
from avalanche.models.base_model import BaseModel


class SimpleMLP(nn.Module, BaseModel):
    """
    Multi-Layer Perceptron with custom parameters.
    It can be configured to have multiple layers and dropout.
    """
    def __init__(self, num_classes=2, input_size=2048,
                 hidden_size=512, hidden_layers=2, drop_rate=0.1):
        """
        :param num_classes: output size
        :param input_size: input size
        :param hidden_size: hidden layer size
        :param hidden_layers: number of hidden layers
        :param drop_rate: dropout rate. 0 to disable
        """
        super().__init__()
        self.hidden_layers = hidden_layers
        layers = nn.Sequential()
        layers.add_module("fc0", nn.Sequential(
            *(nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate))
        ))
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}", nn.Sequential(
                    *(nn.Linear(hidden_size, hidden_size),
                      nn.ReLU(inplace=True),
                      nn.Dropout(p=drop_rate))))

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self._input_size = input_size

    def forward(self, x):
        # x = x.contiguous()
        # x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        return x

class LoRAMLP(nn.Module):
    """
    Multi-Layer Perceptron with custom parameters and LoRA.
    It can be configured to have multiple layers and dropout.
    """
    def __init__(self, num_classes=2, input_size=2048,
                 hidden_size=512, hidden_layers=2, drop_rate=0.1, lora_rank=16):
        """
        :param num_classes: output size
        :param input_size: input size
        :param hidden_size: hidden layer size
        :param hidden_layers: number of hidden layers
        :param drop_rate: dropout rate. 0 to disable
        :param lora_rank: rank for LoRA adaptation matrices
        """
        super().__init__()
        self.hidden_layers = hidden_layers
        layers = nn.Sequential()
        layers.add_module("fc0", nn.Sequential(
            *(lora.Linear(input_size, hidden_size, r=lora_rank),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate))
        ))
        
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}", nn.Sequential(
                    lora.Linear(hidden_size, hidden_size, r=lora_rank),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=drop_rate)
                )
            )

        self.features = nn.Sequential(*layers)
        self.classifier = lora.Linear(hidden_size, num_classes, r=lora_rank)
        self._input_size = input_size

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        return x


class LoRAMLP_withinitweight(nn.Module, BaseModel):
    """
    Multi-Layer Perceptron with custom parameters.
    It can be configured to have multiple layers and dropout.
    """
    def __init__(self, num_classes=2, input_size=2048,
                 hidden_size=512, hidden_layers=2, drop_rate=0.0, lora_rank = 128):
        """
        :param num_classes: output size
        :param input_size: input size
        :param hidden_size: hidden layer size
        :param hidden_layers: number of hidden layers
        :param drop_rate: dropout rate. 0 to disable
        """
        super().__init__()
        self.hidden_layers = hidden_layers
        layers = []
        
        # First layer
        layers.append(nn.Sequential(
            lora.Linear(input_size, hidden_size, r=lora_rank, merge_weights=False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=drop_rate)
        ))
        
        # Hidden layers
        for layer_idx in range(hidden_layers - 1):
            layers.append(nn.Sequential(
                lora.Linear(hidden_size, hidden_size, r=lora_rank, merge_weights=False),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(p=drop_rate)
            ))
        
        self.features = nn.Sequential(*layers)
        
        # Classifier layer
        self.classifier = lora.Linear(hidden_size, num_classes, r=lora_rank, merge_weights=False)
        
        self._input_size = input_size
        self._initialize_weights()
        self.set_requires_grad_true()

    def forward(self, x):
        # x_list = []
        for layer in self.features:
            # x_list.append(x)
            x = layer(x)
        # x_list.append(x)
        x = self.classifier(x)
        return x #, x_list

    def get_features(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear) or isinstance(m, lora.Linear):
    #             # nn.init.constant_(m.weight, 0)  # 初始化权重为全零
    #             nn.init.zeros_(m.weight)
    #             if m.bias is not None:
    #                 # nn.init.constant_(m.bias, 0)  # 初始化偏置为全零
    #                 nn.init.zeros_(m.bias)

    
    def set_requires_grad_true(self):
        for param in self.parameters():
            param.requires_grad = True


class SimpleMLP_withinitweight(nn.Module, BaseModel):
    """
    Multi-Layer Perceptron with custom parameters.
    It can be configured to have multiple layers and dropout.
    """
    def __init__(self, num_classes=2, input_size=2048,
                 hidden_size=512, hidden_layers=2, drop_rate=0.1):
        """
        :param num_classes: output size
        :param input_size: input size
        :param hidden_size: hidden layer size
        :param hidden_layers: number of hidden layers
        :param drop_rate: dropout rate. 0 to disable
        """
        super().__init__()
        self.hidden_layers = hidden_layers
        layers = nn.Sequential()
        layers.add_module("fc0", nn.Sequential(
            *(nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate))
        ))
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}", nn.Sequential(
                    *(nn.Linear(hidden_size, hidden_size),
                      nn.ReLU(inplace=True),
                      nn.Dropout(p=drop_rate))))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(512, num_classes)
        self._input_size = input_size
        self._initialize_weights()

    def forward(self, x):
        # x_list = []
        for layer in self.features:
            # x_list.append(x)
            x = layer(x)
        # x_list.append(x)
        x = self.classifier(x)
        return x #, x_list

    def get_features(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class SimpleMLP_withbninitweight(nn.Module, BaseModel):
    """
    Multi-Layer Perceptron with custom parameters.
    It can be configured to have multiple layers and dropout.
    """
    def __init__(self, num_classes=2, input_size=2048,
                 hidden_size=512, hidden_layers=2, drop_rate=0.1):
        """
        :param num_classes: output size
        :param input_size: input size
        :param hidden_size: hidden layer size
        :param hidden_layers: number of hidden layers
        :param drop_rate: dropout rate. 0 to disable
        """
        super().__init__()
        self.hidden_layers = hidden_layers
        layers = nn.Sequential()
        layers.add_module("fc0", nn.Sequential(
            *(nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate))
        ))
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}", nn.Sequential(
                    *(nn.Linear(hidden_size, hidden_size),
                      nn.BatchNorm1d(hidden_size),
                      nn.ReLU(inplace=True),
                      nn.Dropout(p=drop_rate))))
        layers.add_module(
                f"fc{layer_idx + 2}", nn.Sequential(
                    *(nn.Linear(hidden_size, 256),
                      nn.BatchNorm1d(256),
                      nn.ReLU(inplace=True),
                      nn.Dropout(p=drop_rate))))
        layers.add_module(
                f"fc{layer_idx + 3}", nn.Sequential(
                    *(nn.Linear(256, 128),
                      nn.BatchNorm1d(128),
                      nn.ReLU(inplace=True),
                      nn.Dropout(p=drop_rate))))
        layers.add_module(
                f"fc{layer_idx + 4}", nn.Sequential(
                    *(nn.Linear(128, 64),
                      nn.ReLU(inplace=True),
                      nn.Dropout(p=drop_rate))))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(64, num_classes)
        self._input_size = input_size
        self._initialize_weights()

    def forward(self, x):
        # x = x.contiguous()
        # x = x.view(x.size(0), self._input_size)
        x_list = []
        for layer in self.features:
            x_list.append(x)
            x = layer(x)
        x_list.append(x)
        x = self.classifier(x)
        return x, x_list

    def get_features(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class MTSimpleMLP(MultiTaskModule):
    """Multi-layer perceptron with multi-head classifier"""
    def __init__(self, input_size=28 * 28, hidden_size=512):
        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.classifier = MultiHeadClassifier(hidden_size)
        self._input_size = input_size

    def forward(self, x, task_labels):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        x = self.classifier(x, task_labels)
        return x


__all__ = [
    'SimpleMLP',
    'LoRAMLP',
    'MTSimpleMLP'
]
