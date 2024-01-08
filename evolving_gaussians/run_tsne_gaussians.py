import pickle
import numpy as np
from sklearn.manifold import TSNE
from sta import sta_matrix, sdtw_matrix
from arg_parser import create_tsne_parser  # Ensure arg_parser is available

# change this if you have GPUs
# in our platform, this experiment ran on 4 GPUs in around 20 minutes

n_gpu_devices = 1

if __name__ == "__main__":
    parser = create_tsne_parser()
    parser.add_argument('--speed', type=float, default=1.0, help='Speed of movement (1.0 or -1.0)')
    parser.add_argument('--num_gaussians', type=int, default=1, help='Number of gaussians in the evolving mixture (1, 3, etc.)')
    parser.add_argument('--evolution_type', type=str, choices=['linear', 'circular'], default='linear', help='Type of evolution (linear or circular)')
    args = parser.parse_args()

    # Load the dataset from the file
    dataset_file = args.dataset_file  # Add this command line argument to specify the dataset file
    gaussian_data = np.load(dataset_file)

    num_classes, num_samples_per_class, num_timesteps, support_size = gaussian_data.shape

    # Other command-line arguments
    betas = args.betas
    perplexity = args.perplexity
    speed = args.speed
    num_gaussians = args.num_gaussians
    evolution_type = args.evolution_type

    experiment = dict(
        gaussian_data=gaussian_data,
        betas=betas,
        speed=speed,
        num_gaussians=num_gaussians,
        evolution_type=evolution_type
    )

    n_samples, n_times, dimension = gaussian_data.shape
    print('----Gaussian mixture signals are the training data for t-SNE-----')
    print('n_samples=', n_samples)
    print('n_times=', n_times)
    print('dimension=', dimension)
    print('-------------------------------------')

    params = dict(epsilon=epsilon, gamma=gamma, n_jobs=4, n_gpu_devices=n_gpu_devices)

    # Compute STA distance matrix
    print('-----Starting sta_matrix computation-------')
    precomputed = sta_matrix(gaussian_data, betas, **params)
    print('-----sta_matrix computed -------')
    print('precomputed sta_matrix has shape', precomputed.shape)

    experiment["sta"] = dict()
    for beta, train_ in zip(betas, precomputed):
        train = train_.copy()
        # Shift the distance to avoid negative values with large betas
        train -= train.min()
        tsne_data = TSNE(metric="precomputed", perplexity=perplexity, init='random').fit_transform(train)
        experiment["sta"][beta] = tsne_data

    method = "soft"
    experiment["soft"] = dict()
    for beta in betas:
        precomputed = sdtw_matrix(gaussian_data, beta, n_jobs=10)
        train = precomputed.copy()
        # Shift the distance to avoid negative values with large betas
        train -= train.min()
        tsne_data = TSNE(metric="precomputed", perplexity=perplexity, init='random').fit_transform(train)
        experiment[method][beta] = tsne_data

    expe_filename = f"data/tsne-gaussians_ns{num_samples_per_class}_nt{n_times}_b{'_'.join(map(str, betas))}_p{perplexity}_s{speed}_ng{num_gaussians}_{evolution_type}.pkl"
    expe_file = open(expe_filename, "wb")

    pickle.dump(experiment, expe_file)
