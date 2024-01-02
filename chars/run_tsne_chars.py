import numpy as np
from sklearn.manifold import TSNE
import pickle
import time
import torch
from sta import sta_matrix, sdtw_matrix
from sta.utils import cost_matrix, tonumpy


# change this if you have GPUs
# in our platform, this experiment ran on 4 GPUs in around 8 minutes

n_gpu_devices = 0
perplexity=10 # Must be lower than nb of time series/samples, default value is 30

if __name__ == "__main__":

    t = time.time()
    # 20 samples per letter
    samples_per_letter = 10
    total_letters = 7  # Assuming there are 7 letters
    X = np.concatenate([np.load(f"data/chars-processed.npy")[i * 20:(i * 20) + samples_per_letter] for i in range(total_letters)])
    print('X.shape=', X.shape)

    y = np.concatenate([np.load(f"data/chars-labels-processed.npy")[i * 20:(i * 20) + samples_per_letter] for i in range(total_letters)])
    print('y.shape=', y.shape)
    print('y=', y)

    n_samples, n_times, dimension, _ = X.shape
    epsilon = 1 / dimension
    gamma = 1.

    # create ground metrics. M corresponds to the convolutional distance
    # matrix for convolutional sinkhorn

    _, M = cost_matrix(dimension)
    K = tonumpy(torch.exp(- M / epsilon))

    # betas = [0, 0.001, 0.01, 0.1, 0.5, 1., 2., 3., 5., 10.]
    betas=[0,0.1]
    experiment = dict(X=X, y=y, betas=betas, epsilon=epsilon, gamma=gamma)
    train_data = []
    params = dict(K=K, epsilon=epsilon, gamma=gamma, n_jobs=1,
                  n_gpu_devices=n_gpu_devices)

    # compute sta distance matrix
    print('-----Starting sta_matrix computation-------')
    precomputed = sta_matrix(X, betas, **params)
    print('-----sta_matrix computed -------')

    experiment["sta"] = dict()
    for beta, train_ in zip(betas, precomputed):
        train = train_.copy()
        # shift the distance to avoid negative values
        train -= train.min()
        tsne_data = TSNE(metric="precomputed",perplexity=perplexity,init='random').fit_transform(train)
        experiment["sta"][beta] = tsne_data

    # compute soft-dtw distance matrix
    method = "soft"
    experiment["soft"] = dict()
    for beta in betas:
        precomputed = sdtw_matrix(X, beta, n_jobs=1)
        train = precomputed.copy()
        # shift the distance to avoid negative values
        train -= train.min()
        tsne_data = TSNE(metric="precomputed",perplexity=perplexity,init='random').fit_transform(train)
        experiment[method][beta] = tsne_data

    # save all
    expe_file = open("data/tsne-chars.pkl", "wb")
    pickle.dump(experiment, expe_file)
    t = time.time() - t
    print("Full time: ", t)
