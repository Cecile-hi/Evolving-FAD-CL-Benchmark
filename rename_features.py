import os

def rename_flac_to_npy(directory):
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 构造文件的完整路径
        file_path = os.path.join(directory, filename)
        # 检查文件是否为 .flac 后缀
        if filename.endswith('.flac'):
            # 构造新的文件路径，将 .flac 替换为 .npy
            new_file_path = file_path.replace('.flac', '.npy')
            # 重命名文件
            os.rename(file_path, new_file_path)
            print(f'Renamed: {file_path} to {new_file_path}')

if __name__ == "__main__":
    # 示例：调用函数并传入目标目录路径
    directory_path = '/data1/zhangxiaohui/all_dataset/FAD-CL-Benchmark/features/train_fadcl_wav2vec/5/fake/'  # 替换为目标目录路径
    rename_flac_to_npy(directory_path)
