import argparse
import ast

def create_tsne_parser():
    parser = argparse.ArgumentParser(description="Run t-SNE experiment with evolving Gaussian mixtures.")
    
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to the dataset file (e.g., gaussian_data.npy)")
    parser.add_argument("--betas", type=ast.literal_eval, default=[0, 0.1], help="List of beta values")
    parser.add_argument("--gamma", type=float, default=1.0, help="Unbalanced Wasserstein distance hyperparameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Entropy regularization hyperparameter")
    parser.add_argument("--perplexity", type=int, default=30, help="Perplexity parameter of t-SNE")

    return parser
