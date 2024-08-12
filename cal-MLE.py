import os
import numpy as np
from sklearn.neighbors import NearestNeighbors

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

def estimate_intrinsic_dimension(features, k=10):
    n_samples = features.shape[0]
    
    # Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(features)
    distances, indices = nbrs.kneighbors(features)
    
    # Compute the local intrinsic dimensionality (LID) for each point
    lids = []
    for i in range(n_samples):
        neighbors = distances[i, 1:]  # ignore the distance to itself
        lids.append(estimate_lid(neighbors))
    
    # Average LID to get the global intrinsic dimensionality
    return np.mean(lids)

def estimate_lid(neighbors):
    k = len(neighbors)
    if k <= 1:
        return 0.0
    
    neighbors = np.sort(neighbors)
    r_k = neighbors[-1]
    
    lid = -k / np.sum(np.log(neighbors / r_k))
    return lid

directory = 'features/train_fadcl_wav2vec/'
features = load_features(directory)
d_M = estimate_intrinsic_dimension(features)
print(f'Estimated intrinsic dimension (d_M) using MLE: {d_M}')
