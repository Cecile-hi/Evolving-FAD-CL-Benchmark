import os
import matplotlib.pyplot as plt
import re

def plot_from_txt_files(directory, output_file):
    plt.figure(figsize=(10, 6))
    
    # 遍历指定目录下的所有txt文件
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()
                last_line = lines[-1].strip()
                
                # 提取数据
                match = re.search(r'\[(.*?)\]', last_line)
                if match:
                    data_str = match.group(1)
                    data_list = [float(x) for x in data_str.split(',')]
                    
                    # 横坐标为数据顺序，纵坐标为1减去数据值
                    x = range(1, len(data_list) + 1)
                    y = [1 - value for value in data_list]
                    
                    # 文件名作为曲线名字
                    label = os.path.splitext(filename)[0]
                    
                    # 画图
                    plt.plot(x, y, label=label)
                    
                    for i, value in enumerate(y):
                        plt.text(x[i], y[i], f'{y[i]:.3f}', fontsize=8, ha='right')
    
    # 设置图例、标题和坐标轴标签
    plt.legend()
    plt.title('Compared with all methods')
    plt.xlabel('Task')
    plt.ylabel('1 - EER')
    plt.grid(True)
    
    # 保存图形
    plt.savefig(output_file)
    
    # 显示图形
    plt.show()

# 使用示例
directory_path = 'result/wav2vec/eer'  # 替换为你的目录路径
output_file = 'all_methods.png'  # 你想保存图形的文件名和格式
plot_from_txt_files(directory_path, output_file)
