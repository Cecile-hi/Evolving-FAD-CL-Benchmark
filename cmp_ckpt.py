import os
import torch

def load_models(directory_path, base_model_name, num_models):
    models = []
    for i in range(num_models):
        model_path = os.path.join(directory_path, f"{base_model_name}_{i}.pt")
        if os.path.exists(model_path):
            model = torch.load(model_path)
            models.append(model)
        else:
            print(f"Model file '{model_path}' does not exist.")
            return None
    return models

def compare_model_parameters(models):
    cmp_flag = True
    for i in range(1, len(models)):
        for param1, param2 in zip(models[0].values(), models[i].values()):
            if not torch.allclose(param1, param2, atol=1e-3):  # atol=1e-3 表示比较到小数点后三位
                import pdb; pdb.set_trace()
                print(f"Parameters of model 0 and model {i} are not equal (up to 3 decimal places).")
                cmp_flag = False
                # return False
    if cmp_flag:
        print("All model parameters are equal (up to 3 decimal places).")
    return cmp_flag

# 使用示例
directory_path = "/data1/zhangxiaohui/all_dataset/FAD-CL-Benchmark/lora_cpts"
base_model_name = "base_model"
num_models = 8

models = load_models(directory_path, base_model_name, num_models)
if models is not None:
    compare_model_parameters(models)
