import os
import matplotlib.pyplot as plt
import re

def plot_from_txt_files(directory, output_file):
    plt.figure(figsize=(12, 8))
    
    # 遍历指定目录下的所有txt文件
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()
                
                # 从第二行开始读取
                for i, line in enumerate(lines[1:], start=1):
                    line = line.strip()
                    
                    # 提取数据
                    match = re.search(r'\[(.*?)\]', line)
                    if match:
                        data_str = match.group(1)
                        data_list = [float(x) for x in data_str.split(',')]
                        
                        # 横坐标为数据顺序，纵坐标为1减去数据值
                        x = range(1, len(data_list) + 1)
                        y = [1 - value for value in data_list]
                        
                        # 文件名和行号作为曲线名字
                        label = f"{os.path.splitext(filename)[0]} Task{i}"
                        
                        # 画图
                        plt.plot(x, y, label=label, marker='o')
                        
                        # 显示每个数据点的值
                        for j, value in enumerate(y):
                            plt.text(x[j], y[j], f'{y[j]:.3f}', fontsize=8, ha='right')
    
    # 设置图例、标题和坐标轴标签
    plt.legend()
    plt.title('Naive on each experiences')
    plt.xlabel('Tasks')
    plt.ylabel('1 - EER')
    plt.grid(True)
    
    # 保存图形
    plt.savefig(output_file)
    
    # 显示图形
    plt.show()

def plot_from_txt_file_old(filepath, output_file):
    plt.figure(figsize=(12, 8))
    
    with open(filepath, 'r') as file:
        lines = file.readlines()
        
        # 从第二行开始读取
        for i, line in enumerate(lines[1:], start=1):
            line = line.strip()
            
            # 提取数据
            match = re.search(r'\[(.*?)\]', line)
            if match:
                data_str = match.group(1)
                data_list = [float(x) for x in data_str.split(',')]
                
                # 横坐标为数据顺序，纵坐标为1减去数据值
                x = range(1, len(data_list) + 1)
                y = [1 - value for value in data_list]
                
                # 行号作为曲线名字
                label = f"Task{i}"
                
                # 画图
                plt.plot(x, y, label=label, marker='o')
                
                # 显示每个数据点的值
                for j, value in enumerate(y):
                    plt.text(x[j], y[j], f'{y[j]:.3f}', fontsize=8, ha='right')
    plt.savefig(output_file)

def plot_from_txt_file(filepath, output_file):
    # 创建一个包含7个子图的图形
    fig, axs = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    
    with open(filepath, 'r') as file:
        lines = file.readlines()
        
        # 从第二行开始读取
        for i, line in enumerate(lines[1:], start=1):
            line = line.strip()
            
            # 提取数据
            match = re.search(r'\[(.*?)\]', line)
            if match:
                data_str = match.group(1)
                data_list = [float(x) for x in data_str.split(',')]
                
                # 横坐标为数据顺序，纵坐标为1减去数据值
                x = range(1, len(data_list) + 1)
                y = [1 - value for value in data_list]
                
                # 行号作为曲线名字
                label = f"Task{i}"
                
                # 在第i个子图中画图
                axs[i-1].plot(x, y, label=label, marker='o')
                
                # 显示每个数据点的值
                for j, value in enumerate(y):
                    axs[i-1].text(x[j], y[j], f'{y[j]:.3f}', fontsize=8, ha='right')
                
                # 设置图例、标题和坐标轴标签
                axs[i-1].legend()
                axs[i-1].set_ylabel(f'Task {i}')
                axs[i-1].grid(True)

    # 设置总标题和X轴标签
    fig.suptitle('Naive on each experience')
    plt.xlabel('exper_id')

    # 保存图形
    plt.savefig(output_file)


# 使用示例
directory_path = '/data1/zhangxiaohui/all_dataset/FAD-CL-Benchmark/result/wav2vec/eer/Naive.txt'  # 替换为你的目录路径
output_file = 'naive_plot.png'  # 你想保存图形的文件名和格式
plot_from_txt_file(directory_path, output_file)
