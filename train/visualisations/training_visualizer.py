import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from operator import itemgetter

fontP = FontProperties()
fontP.set_size('small')

class Visualizer():
    def __init__(self, args, total_batches, holdout_size):
        self.args = args
        self.batch_num = total_batches
        self.previous_class_scores = {}
        self.current_class_scores = {}
        self.detailed_micro_scores = [() for i in range(self.batch_num - 1)]
        self.detailed_macro_scores = [() for i in range(self.batch_num - 1)]
        self.incr_state_to_scores = {i: () for i in range(1, self.batch_num)}
        self.holdout_size = holdout_size

    def plot_confusion_matrix(self, df_conf_mat, itera, seed):
        OUT_DIR = 'vis_outputs/'
        out_path = f'corr_vis/by_predictions/{self.args.dataset}_size_{self.holdout_size}_seed_{seed}_itera_{itera}_{self.args.method}_{self.args.exemplar}.png'
        out_path = OUT_DIR + out_path
        f, ax = plt.subplots(1, 1, figsize=(10, 12))
        labels = df_conf_mat.index.tolist()
        hmap = sns.heatmap(df_conf_mat, ax=ax, annot=True, xticklabels=1, yticklabels=1)
        hmap.set_yticklabels(hmap.get_yticklabels(), fontsize=11)
        hmap.set_xticklabels(hmap.get_xticklabels(), fontsize=11)

        for label in ax.get_xticklabels()[1::2]:
            label.set_visible(False)
        fig = hmap.get_figure()
        plt.xlabel("")
        plt.ylabel("")
        plt.tick_params(axis='both', which='major', labelsize=11)
        fig.savefig(out_path, bbox_inches='tight')
