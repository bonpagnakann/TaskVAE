
import torch
from .vae import *
import numpy as np
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, TensorDataset


def sample_within_boundary_box(latent_vectors, num_samples, device):
    num_samples = int(num_samples/2)
    all_random_vecs = []
    all_labels = []
    for label, vectors in latent_vectors.items():
        # Generate a random vector in the range of min and max vectors
        random_vecs = (vectors['max'] - vectors['min']) * torch.rand(num_samples, vectors['min'].size(0), device=device) + vectors['min']

        # Create labels for the generated samples
        labels = torch.full((random_vecs.shape[0],), label, device=device, dtype=torch.long)

        all_random_vecs.append(random_vecs)
        all_labels.append(labels)

    # Concatenate all vectors from all labels along the first dimension
    combined_random_vecs = torch.cat(all_random_vecs, dim=0)
    combined_labels = torch.cat(all_labels, dim=0)

    return combined_random_vecs, combined_labels

def generate_latent_space(vae_model, train_data, mean, std, device):
    vae_model.to(device)
    vae_model.eval()
    
    #with_normalization
    class_dataloaders = {}
    for class_label, data in train_data.items():
        # Assume data is already a tensor of shape (n_samples, channels, features)
        data_tensor = torch.tensor(data)
        normalized_data = (data_tensor - mean[:, None]) / std[:, None]  
        dataset = TensorDataset(normalized_data)  # Only data, labels not needed for reconstruction
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        class_dataloaders[class_label] = data_loader

    # Dictionaries to store the aggregated results
    mean_vectors = {}
    log_var_vectors = {}
    z_vectors = {}

    for label, dataloader in class_dataloaders.items():
        batch_means = []
        batch_log_vars = []
        batch_z = []
        
        for batch in dataloader:
            features = batch[0].float().to(device)  
            
            with torch.no_grad():
                z_mu, z_log_var = vae_model.encode(features)
                z = vae_model.reparameterize(z_mu, z_log_var)
            
            # Collect results for this batch
            batch_means.append(z_mu)
            batch_log_vars.append(z_log_var)
            batch_z.append(z)
        
        # Concatenate all batches and compute the mean of the means and log vars for this class
        class_mean = torch.cat(batch_means)
        class_log_var = torch.cat(batch_log_vars)
        class_z = torch.cat(batch_z)
        
        # Store the results
        mean_vectors[label] = class_mean.cpu().numpy()  
        log_var_vectors[label] = class_log_var.cpu().numpy()
        z_vectors[label] = class_z.cpu().numpy()

    return mean_vectors, log_var_vectors, z_vectors


def select_latent_vectors_by_probability(max_prob, threshold_val=0.6):
    # Filter out samples with confidence scores below threshold
    confidence_threshold = threshold_val
    filtered_indices = (max_prob >= confidence_threshold).nonzero().squeeze()
    filtered_indices = filtered_indices.sort().values

    return filtered_indices

def generate_sample(vae, latent_vecs, sample_size, device, latent_vec_filter = 'none'):

    random_vecs, original_labels = sample_within_boundary_box(latent_vectors=latent_vecs, num_samples=sample_size*400, device=device)

    random_vecs = random_vecs.to(device)

    generated_batch_size = 5000
    max_probabilities_batches = []
    predicted_labels_batches = []
    generated_sample_batches = []

    # 2. Generation Phase: Decode z to generate a sample
    with torch.no_grad():
        for i in range(0, len(random_vecs), generated_batch_size):
            random_vecs_batch = random_vecs[i:i + generated_batch_size].to(device)
            class_logits = vae.classifier(random_vecs_batch)
            # Get the maximum probability and corresponding class label
            max_probabilities, predicted_labels = torch.max(class_logits, dim=1)
            max_probabilities_batches.append(max_probabilities.cpu())
            predicted_labels_batches.append(predicted_labels.cpu())
        # Concatenate all decoded batches
        max_probabilities = torch.cat(max_probabilities_batches, dim=0)
        predicted_labels = torch.cat(predicted_labels_batches, dim=0)

    if latent_vec_filter == 'probability':
        with torch.no_grad():
            for i in range(0, len(random_vecs), generated_batch_size):
                random_vecs_batch = random_vecs[i:i + generated_batch_size].to(device)
                generated_sample = vae.decoder(random_vecs_batch)
                generated_sample_batches.append(generated_sample.cpu())
            # Concatenate all decoded batches
            generated_sample = torch.cat(generated_sample_batches, dim=0)
        selected_indices = select_latent_vectors_by_probability(max_probabilities, 0.6)
        # Access selected data points and labels using the selected indices
        numpy_image = generated_sample[selected_indices]
        numpy_label = original_labels[selected_indices]
        numpy_label = np.array(numpy_label.cpu().numpy())
    elif latent_vec_filter == 'none':
        with torch.no_grad():
            for i in range(0, len(random_vecs), generated_batch_size):
                random_vecs_batch = random_vecs[i:i + generated_batch_size].to(device)
                generated_sample = vae.decoder(random_vecs_batch)
                denormalized_data = generated_sample
                generated_sample_batches.append(denormalized_data.cpu())
            # Concatenate all decoded batches
            numpy_image = torch.cat(generated_sample_batches, dim=0)
        numpy_label = original_labels
        numpy_label = np.array(numpy_label.cpu().numpy())

    numpy_image = numpy_image.squeeze().cpu().numpy()
    
    return numpy_image, numpy_label
