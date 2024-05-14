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
import torch
import pdb
from avalanche.models.dynamic_modules import MultiTaskModule, \
    MultiHeadClassifier
from avalanche.models.base_model import BaseModel



class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1,bias=False)
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask=None):
        """
            N: batch size, T: sequence length, H: Hidden dimension
            input:
                batch_rep : size (N, T, H)
            attention_weight:
                att_w : size (N, T, 1)
            return:
                utter_rep: size (N, H)
        """
        # pdb.set_trace()
        att_logits = self.W(batch_rep).squeeze(-1)
        if att_mask is not None:
            att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = batch_rep * att_w

        return utter_rep, att_w

class CNNSelfAttention(nn.Module):
    def __init__(
        self,
        input_dim = 1024,
        output_class_num = 11
    ):
        super(CNNSelfAttention, self).__init__()
        self.model_seq = nn.Sequential(
            nn.Conv2d(input_dim, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.pooling = SelfAttentionPooling(input_dim = input_dim)
        self.out_layer = nn.Linear(in_features = input_dim, out_features = output_class_num)

    def forward(self, features):

        x_list = []
        # import pdb
        # pdb.set_trace()
        # features = features.transpose(1, 2)
        for i in range (len(self.model_seq)):
            # if i == 0 or i == 2 or i == 4:
                # x_list.append(torch.mean(features, 0, True))
            features = self.model_seq[i](features)
        out = features.transpose(1, 2)
        # x_list.append(torch.mean(features, 0, True))
        # pdb.set_trace()
        out, att_w = self.pooling(features)#.squeeze(-1)
        # x_list.append(torch.mean(features, 0, True))
        out = self.out_layer(features)
        # import pdb
        # pdb.set_trace()
        return out



__all__ = [
    'CNNSelfAttention'
]
