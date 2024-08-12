import os
import torch
import torch.nn as nn
from collections import defaultdict
from scipy.spatial.distance import cosine

# 定义一个简单的示例模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 加载模型参数
def load_model_parameters(checkpoint_path):
    return torch.load(checkpoint_path)

# 计算余弦相似度
def cosine_similarity(tensor1, tensor2):
    return 1 - cosine(tensor1.flatten().cpu(), tensor2.flatten().cpu())

# 比较两个模型参数是否相似
def compare_parameters(params1, params2, threshold=0.99):
    similar_layers = []
    for name in params1:
        if name in params2:
            similarity = cosine_similarity(params1[name], params2[name])
            if similarity > threshold:
                similar_layers.append((name, similarity))
    return similar_layers

# 加载并检查所有的A/B模型参数
def check_model_parameters(base_path, threshold=0.99):
    lora_a_files = [f for f in os.listdir(base_path) if f.startswith('A_')]
    lora_b_files = [f for f in os.listdir(base_path) if f.startswith('B_')]

    lora_a_files.sort()
    lora_b_files.sort()

    a_params_list = [load_model_parameters(os.path.join(base_path, f)) for f in lora_a_files]
    b_params_list = [load_model_parameters(os.path.join(base_path, f)) for f in lora_b_files]

    similar_layers = defaultdict(list)

    for i in range(len(a_params_list)):
        for j in range(i + 1, len(a_params_list)):
            similarities = compare_parameters(a_params_list[i], a_params_list[j], threshold)
            for name, similarity in similarities:
                similar_layers[f"A_{i} & A_{j}"].append((name, similarity))

    for i in range(len(b_params_list)):
        for j in range(i + 1, len(b_params_list)):
            similarities = compare_parameters(b_params_list[i], b_params_list[j], threshold)
            for name, similarity in similarities:
                similar_layers[f"B_{i} & B_{j}"].append((name, similarity))

    return similar_layers

# 检查路径下的所有模型参数
base_path = "/data1/zhangxiaohui/all_dataset/FAD-CL-Benchmark/lora_cpts"
threshold = 0.9  # 设置相似度阈值
similar_layers = check_model_parameters(base_path, threshold)

# 打印结果
for model_pair, layers in similar_layers.items():
    print(f"Model pair {model_pair} has similar layers:")
    for layer, similarity in layers:
        print(f"  Layer {layer} with similarity {similarity:.4f}")
