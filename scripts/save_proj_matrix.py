import os

import numpy as np

from dpipe.io import load, save


embeddings_path = '/shared/experiments/ood_playground/cc359/brain_segm/train_embeddings/experiment_0/'

for i, embedding_file in enumerate(os.listdir(os.path.join(embeddings_path, 'test_predictions'))):
    embeddings = load(os.path.join(embeddings_path, 'test_predictions', embedding_file))
    if i == 0:
        X = embeddings.reshape(embeddings.shape[0], -1).astype(np.float32).T
    else:
        X = np.concatenate([X, embeddings.reshape(embeddings.shape[0], -1).astype(np.float32).T])
    print(f'{i + 1}: Size of X: {(X.size * X.itemsize / 1024 / 1024 / 1024):.4f} Gb', flush=True)

print(X.shape, flush=True)

# normalize
norm = np.linalg.norm(X, axis=-1, keepdims=True)
X /= norm

x_t_x_inv = np.linalg.inv(X.T @ X)

save(x_t_x_inv, os.path.join(embeddings_path, 'projection_matrix.npy'))