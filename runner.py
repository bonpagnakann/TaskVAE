
import argparse
import time
from train import trainer
from config_args import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='motion', type=str,
                    help='Possible values: hhar, motion, uci, motion, realworld')
parser.add_argument('--total_classes', default=6, type=int,
                    help='6 for hhar, motion, and uci, 8 for realworld, 10 for pamap')
parser.add_argument('--new_classes', default=2, type=str, help='number of new classes per incremental batch')
parser.add_argument('--base_classes', default=2, type=int, help='number of classes in first batch')
parser.add_argument('--epochs', default=100, type=int, help='number of training epochs')
parser.add_argument('--T', default=2, type=float, help='temperature value for distillation loss')
parser.add_argument('--average_over', default='holdout', type=str,
                    help="whether to average over different holdout sizes: "
                         " 'holdout', different train percents: 'tp'"
                         "or a single run: 'na'")
parser.add_argument('--method', default='ce', type=str,
                    help="distillation method to use: 'ce' for only cross entropy"
                         "'kd_kldiv' for base distillaiton loss with kl divergence "
                         "'cn_lfc_mr' : cosine norm + less forget constraint + margin ranking loss,"
                         "'ce_ewc': EWC with memory replay,")
parser.add_argument('--exemplar', default='random', type=str, help="exemplar selection strategy: 'random', 'icarl', 'fetril'"
                                                                   "'kmeans', 'dpp', 'boundary', 'sensitivity' or 'fwsr'")
parser.add_argument('--vae_lat_sampling', default='none', type=str, help='Select the sampling strategy for latent vectors from VAE')
parser.add_argument('--latent_vec_filter', default='none', type=str, help='Select the filter strategy for latent vectors from VAE')
parser.add_argument('--person', default=0, type=int, help='Select data of a person to train in CL.')
parser.add_argument('--generated_size', default=1.0, type=float, help='Select the ratio of the average size for generating samples from VAE')
parser.add_argument('--wt_init', default=False, type=bool,
                    help="whether to initialize the weights for old classes using "
                         "data stats or not")
parser.add_argument('--lamda_old', default=2, type=float,
                    help='Base lamda for knowledge distillation loss.')
parser.add_argument('--number', default=0, type=int,
                    help='Experiment Number')
parser.add_argument('--lamda_base', default=2, type=float,
                    help='Base lamda for weighting less forget constraint loss.')
parser.add_argument('--reg_coef', default=5, type=float, help='Regularization coefficient for "online_ewc": a larger '
                                                               'value means less plasticity')
args = parser.parse_args()

if 'ce' not in args.method:
     args.lamda_old, args.lamda_base = get_default_params(args.dataset, args.new_classes, args.method)

def main():
    """
    Main function to train and test.
    """
    start_time = time.time()
    model_trainer = trainer.Trainer(args)
    print(f"Total elapsed time: {time.time() - start_time}")

if __name__ == "__main__":
    main()
