import numpy as np
from sklearn.manifold import TSNE
import pickle
import time
import torch
from sta import sta_matrix, sdtw_matrix
from sta.utils import cost_matrix, tonumpy
import wandb
# from pathlib import Path

# Set to True if you want to log results in wandb and change project destination
wandb=False
# Initialize WandB
if wandb:
    wandb.init(entity="basile-terv", project="spatio-temporal-alignments", name="chars_experiment")


# change this if you have GPUs
# in Hicham Janati's original code, this experiment ran on 4 GPUs in around 8 minutes
n_gpu_devices = 1


if __name__ == "__main__":

    t = time.time()
    samples_per_letter = 5 # choose value up to 20 as there are 20 samples per letter
    total_letters = 7  # Assuming there are 7 letters
    X = np.concatenate([np.load(f"data/chars-processed.npy")[i * 20:(i * 20) + samples_per_letter] for i in range(total_letters)])
    print('X.shape=', X.shape)

    y = np.concatenate([np.load(f"data/chars-labels-processed.npy")[i * 20:(i * 20) + samples_per_letter] for i in range(total_letters)])
    print('y.shape=', y.shape)
    print('y=', y)

    n_samples, n_times, dimension, _ = X.shape
    epsilon = 1 / dimension
    gamma = 1.
    perplexity=10 # Must be lower than n_samples, default value is 30

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

    # Log configuration parameters to WandB
    if wandb:
        config = dict(
            samples_per_letter=samples_per_letter,
            total_letters=total_letters,
            betas=betas,
            epsilon=epsilon,
            gamma=gamma,
            perplexity=perplexity,
            n_gpu_devices=n_gpu_devices,
            # Add other relevant parameters
        )
        wandb.config.update(config)

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
    expe_file_path = "data/tsne-chars.pkl"
    with open(expe_file_path, "wb") as expe_file:
        pickle.dump(experiment, expe_file)
    
    t = time.time() - t
    print("Full time: ", t)

    # Log file to WandB
    if wandb:
        wandb.save(expe_file_path)
