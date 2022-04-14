import os

import numpy as np

from dpipe.io import load, save
# from ood.dataset.luna import LUNA16, get_n_tumors
# from dpipe.split import stratified_train_val_test_split
    

# dataset = LUNA16()
# train_ids = load('configs/assets/cross_val/lidc_train_ids.json')
# n_tumors = get_n_tumors(dataset, train_ids)
# stratified_train_split1 = stratified_train_val_test_split(train_ids, n_tumors,
#                                         val_size=100, n_splits=2, random_state=0)[:1]
# stratified_train_split_tumors = stratified_train_val_test_split(n_tumors, n_tumors,
#                                         val_size=100, n_splits=2, random_state=0)[:1]

# split = stratified_train_val_test_split(stratified_train_split1[0][0], stratified_train_split_tumors[0][0],
#                                         val_size=0, n_splits=3, random_state=0)[:1]
# test_ids = split[0][2]


# embeddings_path = '/shared/experiments/ood_playground/cc359/brain_segm/cc359/train_embeddings/experiment_0/'
# embeddings_path = '/shared/experiments/ood_playground/luna/luna/train_embeddings/experiment_0/'
embeddings_path = '/shared/experiments/ood_playground/midrc/midrc/train_embeddings/experiment_0/'
os.makedirs(os.path.join(embeddings_path, 'stacked_preds'), exist_ok=True)

for fold in range(5):
    for i, embedding_file in enumerate(os.listdir(os.path.join(embeddings_path, 'test_predictions'))[fold * 20 : (fold + 1) * 20]):
        embeddings = load(os.path.join(embeddings_path, 'test_predictions', embedding_file))
        if i == 0:
            X = embeddings.reshape(embeddings.shape[0], -1).astype(np.float32).T
        else:
            X = np.concatenate([X, embeddings.reshape(embeddings.shape[0], -1).astype(np.float32).T])
        print(f'{i + 1}: Size of X: {(X.size * X.itemsize / 1024 / 1024 / 1024):.4f} Gb', flush=True)

    save(X, os.path.join(embeddings_path, f'stacked_preds/{fold}.npy'))

# for i, embedding_file in enumerate(os.listdir(os.path.join(embeddings_path, 'test_predictions'))):
#     embeddings = load(os.path.join(embeddings_path, 'test_predictions', embedding_file))
#     if i == 0:
#         X = embeddings.reshape(embeddings.shape[0], -1).astype(np.float32).T
#     else:
#         X = np.concatenate([X, embeddings.reshape(embeddings.shape[0], -1).astype(np.float32).T])
#     print(f'{i + 1}: Size of X: {(X.size * X.itemsize / 1024 / 1024 / 1024):.4f} Gb', flush=True)

# for i, embedding_file in enumerate(os.listdir(os.path.join(embeddings_path, 'stacked_preds'))):
#     embeddings = load(os.path.join(embeddings_path, 'stacked_preds', embedding_file)).astype(np.float32)
#     print(embeddings.shape, flush=True)
#     if i == 0:
#         X = embeddings
#     else:
#         X = np.concatenate([X, embeddings])
#     print(f'{i + 1}: Size of X: {(X.size * X.itemsize / 1024 / 1024 / 1024):.4f} Gb', flush=True)

for i, embedding_file in enumerate(os.listdir(os.path.join(embeddings_path, 'stacked_preds'))):
    X = load(os.path.join(embeddings_path, 'stacked_preds', embedding_file)).astype(np.float32)
    print(f'{i + 1}: Size of X: {(X.size * X.itemsize / 1024 / 1024 / 1024):.4f} Gb', flush=True)

    print(X.shape, flush=True)

    # normalize
    norm = np.linalg.norm(X, axis=-1, keepdims=True)
    X /= norm

    x_t_x_inv = np.linalg.inv(X.T @ X)

    print(os.path.join(embeddings_path, f'projection_matrix{i}.npy'), flush=True)
    print(x_t_x_inv, flush=True)
    print(flush=True)
    save(x_t_x_inv, os.path.join(embeddings_path, f'projection_matrix{i}.npy'))
    if i == 0:
        X_full = X
    else:
        X_full = np.concatenate([X_full, X])
        
print(X_full.shape, flush=True)

# normalize
norm = np.linalg.norm(X_full, axis=-1, keepdims=True)
X_full /= norm

x_t_x_inv = np.linalg.inv(X_full.T @ X_full)

save(x_t_x_inv, os.path.join(embeddings_path, f'projection_matrix.npy'))

# print(X.shape, flush=True)

# # normalize
# norm = np.linalg.norm(X, axis=-1, keepdims=True)
# X /= norm

# x_t_x_inv = np.linalg.inv(X.T @ X)

# save(x_t_x_inv, os.path.join(embeddings_path, f'projection_matrix.npy'))