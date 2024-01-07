import argparse

def create_tsne_parser():
    parser = argparse.ArgumentParser(description="Run TSNE experiment with specified parameters.")
    parser.add_argument("--n_samples_per_task", type=int, default=20, help="Number of samples per task")
    parser.add_argument("--n_times", type=int, default=10, help="Number of time points")
    parser.add_argument("--time0", type=int, default=3, help="First peak time")
    parser.add_argument("--time1", type=int, default=8, help="Second peak time")
    parser.add_argument("--betas", nargs="*", type=float, default=[0, 0.1], help="List of beta values")
    parser.add_argument("--gamma", type=float, default=1., help="Unbalanced Wasserstein distance hyperparameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Entropy regularization hyperparameter")
    parser.add_argument("--perplexity", type=int, default=30, help="Perplexity parameter of t-SNE")
    return parser
