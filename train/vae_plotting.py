
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from PIL import Image
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans


def plot_losses_and_accuracy(labels, seed, reconstruction_losses, kl_losses, classification_losses, combined_losses, accuracies,
                             val_accuracies, val_losses, val_recon_loss):
    epochs = range(1, len(reconstruction_losses) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, reconstruction_losses, label='Reconstruction Training Loss')
    plt.plot(epochs, val_recon_loss, label='Reconstruction Validation Loss')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig_name = './vis_vae/' + str(seed)+ '/recon_loss_' + str(labels[0]) + '_' + str(labels[1]) + '.png'
    plt.savefig(fig_name)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, kl_losses, label='KL Loss')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig_name = './vis_vae/' + str(seed)+ '/kl_loss_' + str(labels[0]) + '_' + str(labels[1]) + '.png'
    plt.savefig(fig_name)


    plt.figure(figsize=(8, 6))
    plt.plot(epochs, classification_losses, label='Classification Loss')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig_name = './vis_vae/' + str(seed)+ '/classification_loss_' + str(labels[0]) + '_' + str(labels[1]) + '.png'
    plt.savefig(fig_name)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, combined_losses, label='Combined Loss')
    plt.title('Combined Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig_name = './vis_vae/' + str(seed)+ '/combined_loss_' + str(labels[0]) + '_' + str(labels[1]) + '.png'
    plt.savefig(fig_name)


    plt.figure(figsize=(8, 6))
    plt.plot(epochs, accuracies, label='Classifier Accuracy')
    plt.title('Classifier Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig_name = './vis_vae/' + str(seed)+ '/classifier_training_acc_' + str(labels[0]) + '_' + str(labels[1]) + '.png'
    plt.savefig(fig_name)

def plot_accuracy(train_acc):
    num_epochs = len(train_acc)

    plt.plot(np.arange(1, num_epochs + 1),
             train_acc, label='Training')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()


def plot_latent_space(data_loader, model, n_class):
    train_loader = data_loader
    all_labels = []
    all_embeddings = []

    with torch.no_grad():
        for features, targets in train_loader:
            embeddings, z_mean, z_log_var = model.encoding_fn(features)
            all_embeddings.append(embeddings)
            all_labels.append(targets)

    # Concatenate all retrieved embeddings & labels
    final_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    final_labels = torch.cat(all_labels, dim=0).numpy()

    n_clusters = n_class  # Define the number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=123)
    kmeans.fit(final_embeddings)
    cluster_labels = kmeans.labels_


    tsne = TSNE(n_components=3, random_state=123)
    tsne_results = tsne.fit_transform(final_embeddings)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for cluster in range(n_clusters):
        indices = cluster_labels == cluster
        ax.scatter(tsne_results[indices, 0], tsne_results[indices, 1], tsne_results[indices, 2], label=f'Cluster {cluster}', alpha=1)
    ax.set_title("VAE Latent Space with K-Means Clusters (3D t-SNE visualization)")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_zlabel("t-SNE Dimension 3")
    ax.legend()
    plt.show()