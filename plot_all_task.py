import ast
import matplotlib.pyplot as plt
import os
import re
import numpy as np
from scipy.interpolate import make_interp_spline
def read_data_from_txt(file_path):
    data_dict = {}
    with open(file_path, 'r') as file:
        for task_name, line in enumerate(file):
            if line.startswith('Train after'):
                # 提取任务名称
                # task_name += 1  # 提取 "TaskX" 部分
                # 使用正则表达式提取方括号中的列表
                data = re.findall(r'\[([^\]]+)\]', line)[0]
                # 将字符串形式的数字列表转换为浮点数列表
                data = list(map(float, data.split(',')))
                data_dict[task_name] = data
    # import pdb; pdb.set_trace()
    return data_dict


def plot_eer_dict(data_dict, output_dir):
    
    list_length = len(list(data_dict.values())[0])

    os.makedirs(output_dir, exist_ok=True)

    # 准备 x 轴的标签
    x = np.arange(len(data_dict))
    x_labels = list(data_dict.keys())

    # 创建一张包含所有曲线的图
    plt.figure()

    for i in range(list_length):
        y = [values[i] * 100 for values in data_dict.values()]

        # 使用插值平滑曲线
        x_new = np.linspace(x.min(), x.max(), 100)  # 创建更多点以便平滑
        spl = make_interp_spline(x, y, k=1)  # 使用1次样条插值
        y_smooth = spl(x_new)

        # 画平滑的曲线
        plt.plot(x_new, y_smooth, label=f'Eval on Task {i+1}', linewidth=2)

        # 在原始点上添加标记
        plt.scatter(i, y[i], marker='o', s=50, zorder=7)  # 只在对应的任务上标记

    # 设置图表属性
    plt.xticks(ticks=x, labels=x_labels)
    
    plt.xlabel('After Training on Tasks')
    plt.ylabel('EER (%)')
    plt.title('Evaluation EER on All Tasks')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'Evaluation_on_all_tasks.pdf'))
    plt.close()

# 指定文件路径
file_path = '/data1/zhangxiaohui/all_dataset/FAD-CL-Benchmark/result/wav2vec_InitA_100epoch_lora_32_normal_moe/eer/ELMA_r_32.txt'
output_dir = './'

# 读取文件数据并转换为字典
data_dict = read_data_from_txt(file_path)

# 绘制包含所有曲线的图
plot_eer_dict(data_dict, output_dir)
