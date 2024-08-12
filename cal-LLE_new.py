import os
import numpy as np
from sklearn.manifold import LocallyLinearEmbedding

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

def estimate_intrinsic_dimension_lle(features, n_neighbors=10, n_jobs=-1):
    lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, method='standard', n_jobs=n_jobs)
    lle.fit(features)
    
    reconstruction_error = lle.reconstruction_error_
    d_M = reconstruction_error
    return d_M

if __name__ == "__main__":

    d_M_list = []
    for i in range(1,8):
        directory = 'features/train_fadcl_wav2vec/{}'.format(i)
        features = load_features(directory)
        d_M = estimate_intrinsic_dimension_lle(features, n_neighbors=10, n_jobs=-1)
        d_M_list.append(d_M)
    d_M = sum(d_M_list)/7
    print(f'Estimated intrinsic dimension (d_M) using LLE: {d_M}')
