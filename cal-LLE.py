import os
import numpy as np
from sklearn.manifold import LocallyLinearEmbedding

def load_features(directory):
    features = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):  # 假设特征文件是.npy格式
                filepath = os.path.join(root, file)
                feature = np.load(filepath)
                if feature.shape[1] == 1024:  # 确保每个特征长度为1024
                    features.append(feature)
                else:
                    import pdb
                    pdb.set_trace()
    return np.vstack(features)  # 确保返回二维数组

def estimate_intrinsic_dimension_lle(features, n_neighbors=10):
    lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, method='standard')
    lle.fit(features)
    
    reconstruction_error = lle.reconstruction_error_
    d_M = reconstruction_error
    return d_M

directory = 'features/train_fadcl_wav2vec/'
features = load_features(directory)
d_M = estimate_intrinsic_dimension_lle(features)
print(f'Estimated intrinsic dimension (d_M) using LLE: {d_M}')
