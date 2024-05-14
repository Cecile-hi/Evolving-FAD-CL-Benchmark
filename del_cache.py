import os
import shutil

def delete_pycache(directory):
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            if dir == "__pycache__":
                pycache_dir = os.path.join(root, dir)
                shutil.rmtree(pycache_dir)
                print(f"Deleted {pycache_dir}")

# 删除指定目录下和其子目录下的所有 __pycache__ 目录
delete_pycache("/data1/zhangxiaohui/all_dataset/FAD-CL-Benchmark")
