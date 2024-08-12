import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

def load_features(directory):
    features = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):
                filepath = os.path.join(root, file)
                feature = np.load(filepath)
                if feature.shape[1] == 1024:
                    features.append(feature)
    return np.vstack(features)

def estimate_intrinsic_dimension(features, k=10, n_jobs=-1):
    n_samples = features.shape[0]
    
    # Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', n_jobs=n_jobs).fit(features)
    distances, indices = nbrs.kneighbors(features)
    
    # Compute the local intrinsic dimensionality (LID) for each point in parallel
    lids = Parallel(n_jobs=n_jobs)(delayed(estimate_lid)(distances[i, 1:]) for i in range(n_samples))
    
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

# directory = '/path/to/features'
# features = load_features(directory)
# d_M = estimate_intrinsic_dimension(features, k=10, n_jobs=-1)
# print(f'Estimated intrinsic dimension (d_M) using MLE: {d_M}')


if __name__ == "__main__":

    d_M_list = []
    for i in range(1,8):
        directory = 'features/train_fadcl_wav2vec/{}'.format(i)
        features = load_features(directory)
        d_M = estimate_intrinsic_dimension(features, k=10, n_jobs=-1)
        d_M_list.append(d_M)
    d_M = sum(d_M_list)/7
    print(f'Estimated intrinsic dimension (d_M) using MLE: {d_M}')