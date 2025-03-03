
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



def sample_within_adaptive_boundaries(latent_vectors, num_samples, device, lower_percentile=0.05, upper_percentile=0.95):
    # Get the min and max vectors consisting of min and max value of each dimension
    min_vals = torch.quantile(latent_vectors, lower_percentile, dim=0).to(device)
    max_vals = torch.quantile(latent_vectors, upper_percentile, dim=0).to(device)

    # Generate a random vector in the range of min and max vectors
    random_vecs = (max_vals - min_vals) * torch.rand(num_samples, min_vals.size(0),device=device) + min_vals

    return random_vecs

def sample_within_boundary_box(latent_vectors, num_samples, device):
    num_samples = int(num_samples/2)
    all_random_vecs = []
    all_labels = []
    for label, vectors in latent_vectors.items():
        print('Label: ', label)

        print('latent_vectors.keys():', vectors.keys())

        print('latent_vectors[min].shape:', vectors['min'].shape)
        print('latent_vectors[max].shape', vectors['max'].shape)

        print('latent_vectors[min]:', vectors['min'])
        print('latent_vectors[max]', vectors['max'])

        # Generate a random vector in the range of min and max vectors
        random_vecs = (vectors['max'] - vectors['min']) * torch.rand(num_samples, vectors['min'].size(0), device=device) + vectors['min']

        # Create labels for the generated samples
        labels = torch.full((random_vecs.shape[0],), label, device=device, dtype=torch.long)

        #random_vecs =  random_z_mu + torch.randn(num_samples, vectors['min'].size(0), device=device) * torch.exp(vectors['mean_log_var'] / 2)
        print('random_vecs:', random_vecs)

        all_random_vecs.append(random_vecs)
        all_labels.append(labels)

        print('random_vecs[label].shape:', random_vecs[label].shape)
    # Concatenate all vectors from all labels along the first dimension
    combined_random_vecs = torch.cat(all_random_vecs, dim=0)
    combined_labels = torch.cat(all_labels, dim=0)
    print('combined_random_vecs.shape:', combined_random_vecs.shape)

    return combined_random_vecs, combined_labels

def sample_within_boundary_box_in_clusters(latent_vectors, num_samples, device):
    all_samples = []
    all_labels = []

    # Loop over the items in min_max_vectors to handle multiple labels
    for label, vectors in latent_vectors.items():
        print(f"Processing label: {label}")
        
        centroids = vectors['centroids']
        weights = vectors['weights']
        mean_log_var = vectors['mean_log_var']
        
        #print(f"Centroids shape for label {label}: {centroids.shape}")
        #print(f"Mean log variance for label {label}: {mean_log_var}")

        num_centroids = centroids.shape[0]
        #samples_per_centroid = num_samples // num_centroids
        
        print(f"Number of centroids for label {label}: {num_centroids}")
        #print(f"Samples per centroid for label {label}: {samples_per_centroid}")

        # To hold all the generated samples for this label
        sampled_vectors = []
        weights = weights / np.sum(weights)

        for i, centroid in enumerate(centroids):
            # Generate samples_per_centroid samples around this centroid
            samples_per_centroid = int(np.round(weights[i] * num_samples))
            print(f"Samples per centroid for label {label}: {samples_per_centroid}")
            noise = torch.randn(samples_per_centroid, centroids.shape[1]).to(device) * torch.exp(mean_log_var / 2)
            samples = torch.tensor(centroid).to(dtype=torch.float32, device=device) + noise
            sampled_vectors.append(samples)
            
            #print(f"Centroid {i} for label {label}: {centroid}")
            #print(f"Noise shape: {noise.shape}")
            #print(f"Samples shape: {samples.shape}")

        # Concatenate all samples from each centroid
        sampled_vectors = torch.cat(sampled_vectors, dim=0)
        print(f"Sampled vectors shape for label {label}: {sampled_vectors.shape}")

        # Create labels for the generated samples
        labels = torch.full((sampled_vectors.shape[0],), label, device=device, dtype=torch.long)

        # Append to all samples
        all_samples.append(sampled_vectors)
        all_labels.append(labels)

    # Concatenate samples for all classes
    combined_samples = torch.cat(all_samples, dim=0)
    combined_labels = torch.cat(all_labels, dim=0)
    print(f"Combined samples shape: {combined_samples.shape}")
    
    return combined_samples, combined_labels

def sample_by_gmm(latent_vectors, num_samples, device):
    all_samples = []
    all_labels = []

    # Loop over the items in min_max_vectors to handle multiple labels
    for label, vectors in latent_vectors.items():
        print(f"Processing label: {label}")
        
        gmm = vectors['gmm']
        pca = vectors['pca']
        
        # Generate samples from the GMM
        gen_samples, _ = gmm.sample(num_samples)
        # Inverse transform the PCA-reduced samples to original feature space
        sampled_vectors_np = pca.inverse_transform(gen_samples)
        sampled_vectors = torch.from_numpy(sampled_vectors_np).to(dtype=torch.float32)
        print(f"Sampled vectors shape for label {label}: {sampled_vectors.shape}")

        # Create labels for the generated samples
        labels = torch.full((sampled_vectors.shape[0],), label, device=device, dtype=torch.long)

        # Append to all samples
        all_samples.append(sampled_vectors)
        all_labels.append(labels)

    # Concatenate samples for all classes
    combined_samples = torch.cat(all_samples, dim=0)
    combined_labels = torch.cat(all_labels, dim=0)
    print(f"Combined samples shape: {combined_samples.shape}")
    
    return combined_samples, combined_labels


def sample_within_boundary_box_in_clusters_prob(latent_vectors, model, num_samples, prob, device):
    all_samples = []
    all_labels = []

    model.to(device)
    model.eval()

    # Loop over the items in min_max_vectors to handle multiple labels
    for label, vectors in latent_vectors.items():
        print(f"Processing label: {label}")
        
        centroids = vectors['centroids']
        weights = vectors['weights']
        mean_log_var = vectors['mean_log_var']
        
        #print(f"Centroids shape for label {label}: {centroids.shape}")
        #print(f"Mean log variance for label {label}: {mean_log_var}")

        num_centroids = centroids.shape[0]
        #samples_per_centroid = num_samples // num_centroids
        
        print(f"Number of centroids for label {label}: {num_centroids}")
        #print(f"Samples per centroid for label {label}: {samples_per_centroid}")

        # To hold all the generated samples for this label
        sampled_vectors = []
        weights = weights / np.sum(weights)
        print("Weights of each centroid:", weights)

        for i, centroid in enumerate(centroids):
            # Generate samples_per_centroid samples around this centroid
            samples_per_centroid = int(np.round(weights[i] * num_samples))
            print(f"Samples per centroid for label {label}: {samples_per_centroid}")
            noise = torch.randn(samples_per_centroid, centroids.shape[1]).to(device) * torch.exp(mean_log_var / 2)
            samples = torch.tensor(centroid).to(dtype=torch.float32, device=device) + noise
            sampled_vectors.append(samples)
            
            #print(f"Centroid {i} for label {label}: {centroid}")
            #print(f"Noise shape: {noise.shape}")
            #print(f"Samples shape: {samples.shape}")

        # Concatenate all samples from each centroid
        sampled_vectors = torch.cat(sampled_vectors, dim=0)
        print(f"Sampled vectors shape for label {label}: {sampled_vectors.shape}")

        # Create labels for the generated samples
        labels = torch.full((sampled_vectors.shape[0],), label, device=device, dtype=torch.long)

        # Append to all samples
        all_samples.append(sampled_vectors)
        all_labels.append(labels)

    # Concatenate samples for all classes
    combined_samples = torch.cat(all_samples, dim=0)
    combined_labels = torch.cat(all_labels, dim=0)
    print(f"Combined samples shape: {combined_samples.shape}")
    
    # Now filter the combined samples based on the probability threshold
    filtered_samples = []
    filtered_labels = []
    class_sample_counts = {}
    
    generated_batch = 50
    max_probs_batch = []

    with torch.no_grad():
        for i in range(0, len(sampled_vectors), generated_batch):
            sampled_vec_batch = combined_samples[i:i + generated_batch].to(device)
            class_logits = model.classifier(sampled_vec_batch)
            max_prob, predicted_classes = torch.max(class_logits, dim=1)
            max_probs_batch.append(max_prob)

    # Get the maximum probability and corresponding class label
    max_probs = torch.cat(max_probs_batch, dim=0)
    print('max_probs:', max_probs)
    selected_indices = (max_probs >= prob).nonzero().squeeze()
    filtered_samples = combined_samples[selected_indices]
    filtered_labels = combined_labels[selected_indices]

    # Count the number of samples by class after filtering
    for label in combined_labels.unique():
        count = (filtered_labels == label).sum().item()
        class_sample_counts[label.item()] = count

    print(f"Number of samples before filtering: {combined_samples.size(0)}")
    print(f"Number of samples after filtering: {filtered_samples.size(0)}")
    print(f"Sample counts by class after filtering: {class_sample_counts}")

    return combined_samples, combined_labels


def comparison_vae(vae_model, train_data, device, mean=None, std=None):
    vae_model.to(device)
    vae_model.eval()

    data_loaders = {}
    for class_label, data in train_data.items():
        # Assume data is already a tensor of shape (n_samples, channels, features)
        data_tensor = torch.tensor(data)
        dataset = TensorDataset(data_tensor)  # Only data, labels not needed for reconstruction
        data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        data_loaders[class_label] = data_loader


    data_loaders_norm = {}
    for class_label, data in train_data.items():
        # Assume data is already a tensor of shape (n_samples, channels, features)
        data_tensor = torch.tensor(data)
        normalized_data = (data_tensor - mean[:, None]) / std[:, None]  
        dataset = TensorDataset(normalized_data)  # Only data, labels not needed for reconstruction
        data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        data_loaders_norm[class_label] = data_loader
    

    reconstructions = {}

    with torch.no_grad():  # No gradients needed
        for class_label, data_loader in data_loaders_norm.items():
            all_reconstructions = []
            for batch in data_loader:
                data = batch[0].to(device)  # Assuming data is the only element in batch
                reconstructed,_,_,_ = vae_model(data)
                all_reconstructions.append(reconstructed)

            # Combine all batches into a single tensor
            reconstructions[class_label] = torch.cat(all_reconstructions, dim=0)

    denormalized_reconstructions = {}

    for class_label, data in reconstructions.items():
        denormalized_data = data * std.view(1, 6, 1).to(device) + mean.view(1, 6, 1).to(device)
        denormalized_reconstructions[class_label] = denormalized_data
        print(f"Class {class_label} - Denormalized data shape: {denormalized_data.shape}")


    min_max_values = {}
    for class_label, data in denormalized_reconstructions.items():
        # Calculate min and max along the batch dimension (dim=0)
        min_values = torch.min(data, dim=0)[0]
        max_values = torch.max(data, dim=0)[0]
        
        # Store the results in a dictionary
        min_max_values[class_label] = {
            'min': min_values,
            'max': max_values
        }
        
        # Optionally, print the min and max for each class
        print(f"Class {class_label} - Min: {min_values}, Max: {max_values}")

    
    for class_label, data in data_loaders.items():
        # Calculate min and max along the batch dimension (dim=0)
        
        for batch in data:
            data = batch[0].to(device)
            min_values = torch.min(data, dim=0)[0]
            max_values = torch.max(data, dim=0)[0]
                
        # Optionally, print the min and max for each class
        print(f"Class Train Data {class_label} - Min: {min_values}, Max: {max_values}")
        
    return min_max_values


def generate_latent_space(vae_model, train_data, mean, std, device):
    vae_model.to(device)
    vae_model.eval()
    '''no_normalization
    # Create a dictionary of DataLoaders
    class_dataloaders = {}
    for label, data in train_data.items():
        data_tensor = torch.tensor(data, dtype=torch.float32)
        dataset = TensorDataset(data_tensor) 
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        class_dataloaders[label] = dataloader
    '''
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
            features = batch[0].float().to(device)  # Move batch to the same device as the model
            
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
        mean_vectors[label] = class_mean.cpu().numpy()  # Optionally convert to numpy
        log_var_vectors[label] = class_log_var.cpu().numpy()
        z_vectors[label] = class_z.cpu().numpy()
        
        #print(f"Label: {label}")
        #print("Shape of mean vector:", mean_vectors[label].shape)
        #print("Shape of log variance vector:", log_var_vectors[label].shape)

    return mean_vectors, log_var_vectors, z_vectors

    #return latent_vecs, class_label


def sample_from_latent_space_gmm(lat_vecs, class_labels, num_samples):
    vecs = lat_vecs.detach().cpu().numpy()
    labels = class_labels.detach().cpu().numpy()
    print('random_vecs.shape:', vecs.shape)
    print('class_labels.shape:', class_labels.shape)

    # Get unique labels
    unique_labels = np.unique(labels)

    min_components = 5
    max_components = 10

    all_samples = []

    # Iterate over each unique label
    for label in unique_labels:
        # Get indices corresponding to the current label
        label_indices = np.where(labels == label)[0]
        #print('indices for label', label, ':', label_indices)
        print('len(label_indices):', len(label_indices))
        # Get data points for the current label
        label_data = vecs[label_indices]

        best_score = -1
        best_model = None
        best_num_components = -1
        lowest_bic = np.infty

        for n_components in range(min_components, max_components + 1):
            # Fit Gaussian Mixture Model with specified number of components
            gmm = GaussianMixture(n_components=n_components)
            gmm.fit(label_data)
            bic = gmm.bic(label_data)

            #print('bic for n_components = ', n_components, 'is:', bic)
            if bic < lowest_bic:
                lowest_bic = bic
                best_gmm = gmm
                best_num_components = n_components

        print('best_num_components for label ', label, 'is', best_num_components)
        samples, _ = best_gmm.sample(num_samples)
        print('samples.shape:', samples.shape)
        all_samples.extend(samples)

        # Get cluster assignments for each data point
        cluster_labels = best_gmm.predict(label_data)

        # Perform PCA for dimensionality reduction to 2 dimensions
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(label_data)

        # Plot the data points with different colors for each cluster
        plt.figure(figsize=(8, 6))
        for label in np.unique(cluster_labels):
            plt.scatter(data_pca[cluster_labels == label, 0], data_pca[cluster_labels == label, 1], label=f'Cluster {label}')

        plt.title('PCA Visualization of GMM Clusters')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True)
        plt.show()

    all_samples = np.array(all_samples)
    print('all_samples.shape:', all_samples.shape)
    all_samples = torch.tensor(all_samples, dtype=torch.float32)
    return all_samples

def cluster_latent_space_kmeans(lat_vecs, class_labels, num_clusters):

    vecs = lat_vecs.detach().cpu().numpy()
    labels = class_labels.detach().cpu().numpy()
    print('lat_vecs.shape:', vecs.shape)
    print('labels.shape:', labels.shape)

    # Get unique labels
    unique_labels = np.unique(labels)

    # Initialize a dictionary to store mean distances for each centroid
    centroid_mean_distances = {}
    cluster_centroids = {}

    # Iterate over each unique label
    for label in unique_labels:
        # Get indices corresponding to the current label
        label_indices = np.where(labels == label)[0]

        # Get data points for the current label
        label_data = vecs[label_indices]

        # Perform KMeans clustering with specified number of clusters
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(label_data)

        # Get the labels of the assigned clusters for each data point
        cluster_labels = kmeans.labels_

        # Initialize a dictionary to store distances for each centroid
        centroid_distances = {i: [] for i in range(num_clusters)}

        #add centroids according to their label
        cluster_centroids[label] = kmeans.cluster_centers_

        # Calculate distances of each data point to its assigned centroid
        for i, centroid in enumerate(kmeans.cluster_centers_):
            cluster_data = label_data[cluster_labels == i]
            distances = np.linalg.norm(cluster_data - centroid, axis=1)
            centroid_distances[i] = distances

        # Compute mean distance for each centroid within the class
        for centroid, distances in centroid_distances.items():
            mean_distance = np.mean(distances)
            centroid_mean_distances[(label, centroid)] = mean_distance

    # Print the mean distances for each centroid within each class
    for (label, centroid), mean_distance in centroid_mean_distances.items():
        print(f"Mean distance for centroid {centroid} in class {label}: {mean_distance}")

    #print('cluster_centroids: ', cluster_centroids)

    return centroid_mean_distances , cluster_centroids

def select_latent_vectors_by_distance(vecs, labels, mean_distance_centroids, centroids):
    vecs = vecs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    print('random_vecs.shape:', vecs.shape)

    # Get unique labels
    unique_labels = np.unique(labels)

    numpy_img = []
    numpy_class = []

    distance_indices = {}
    print('unique_labels:', unique_labels)
    # Iterate over each unique label
    for label in unique_labels:
        # Get indices corresponding to the current label
        label_indices = np.where(labels == label)[0]

        # Get data points for the current label
        label_data = vecs[label_indices]

        # Iterate over each latent vector
        print('label:', label)
        for latent_vec in label_data:
            distances_to_centroids = []
            all_mean_distance = []
            for (l, i), mean_distance in mean_distance_centroids.items():
                if l==label:
                    # Compute distances to each centroid
                    d = np.linalg.norm(latent_vec - centroids[l][i])
                    #print('latent_vec: ', latent_vec)
                    #print('centroid [', l, '][', i, ']: ', centroids[l][i])
                    distances_to_centroids.append(d)
                    all_mean_distance.append(mean_distance)
            # Find the index of the closest cluster centroid
            closest_cluster_index = np.argmin(distances_to_centroids)
            #print('closest centroid distance:', distances_to_centroids[closest_cluster_index])

            if distances_to_centroids[closest_cluster_index] < all_mean_distance[closest_cluster_index]:
                #print('accepted')
                #print('centroid', i, 'distance:', d)
                #print('selected centroid: ', closest_cluster_index, 'value:', distances_to_centroids[closest_cluster_index])
                #print('mean distance:', all_mean_distance[closest_cluster_index])
                numpy_img.append(latent_vec)
                numpy_class.append(label)

    numpy_img = np.array(numpy_img)
    numpy_class = np.array(numpy_class)
    '''
    # Print the mean distances for each centroid within each class
    for (label, centroid), mean_distance in mean_distance_centroids.items():
        print(f"Mean distance for centroid {centroid} in class {label}: {mean_distance}")
    print('numpy_img.shape:', numpy_img.shape)
    print('numpy_class.shape:', numpy_class.shape)
    '''

    print('np.unique(numpy_class):', np.unique(numpy_class))
    return numpy_img, numpy_class

def select_latent_vectors_by_kmeans(vecs, labels, num_clusters, num_near_samples):
    vecs = vecs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    print('random_vecs.shape:', vecs.shape)

    # Get unique labels
    unique_labels = np.unique(labels)

    # Initialize dictionary to store selected indices for each class
    selected_indices = {}

    # Iterate over each unique label
    for label in unique_labels:
        # Get indices corresponding to the current label
        label_indices = np.where(labels == label)[0]

        # Get data points for the current label
        label_data = vecs[label_indices]

        label_data_indices = [each for each in range(len(label_data))]
        label_map = dict(zip(label_data_indices, label_indices))

        # Perform KMeans clustering with 6 clusters
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(label_data)

        # Initialize list to store selected indices for each cluster
        cluster_indices = []

        # Iterate over each cluster
        for centroid in kmeans.cluster_centers_:
            # Find the nearest data points to the centroid of the current cluster
            distances = np.linalg.norm(label_data - centroid, axis=1)  # Compute Euclidean distances
            cluster_indices.extend(np.argsort(distances)[:num_near_samples])  # Select 600 nearest indices

        cluster_indices.sort()

        selected_indices[label] = [label_map[idx] for idx in cluster_indices]
        print("Selected values from cluster indices:", cluster_indices)
        print("Selected values from original indices:", selected_indices[label])

    # Print selected indices for each class
    for label, indices in selected_indices.items():
        print(f"Class {label}: {indices}")

    all_selected_indices = np.concatenate(list(selected_indices.values()))
    return all_selected_indices


def select_latent_vectors_by_probability(max_prob, threshold_val=0.95):
    # Filter out samples with confidence scores below threshold
    print('threshold_val:', threshold_val)
    confidence_threshold = threshold_val
    filtered_indices = (max_prob >= confidence_threshold).nonzero().squeeze()
    filtered_indices = filtered_indices.sort().values

    return filtered_indices

'''
    # Define the ranges
    ranges = [(0.0, 0.50), (0.50, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 0.95), (0.95, 0.97), (0.97, 0.99),
              (0.99, 1.1)]

    # Initialize dictionaries to store class frequencies in each range
    class_frequencies = {range_: {} for range_ in ranges}

    # Count the number of values and class frequencies in each range
    for range_ in ranges:
        min_val, max_val = range_
        range_mask = ((max_probabilities >= min_val) & (max_probabilities < max_val))
        values_in_range = range_mask.sum().item()
        classes_in_range = predicted_labels[range_mask]
        class_counts = torch.bincount(classes_in_range, minlength=class_logits.size(1))
        class_frequencies[range_]['values'] = values_in_range
        class_frequencies[range_]['class_counts'] = class_counts

    # Print the results
    for range_ in ranges:
        min_val, max_val = range_
        print(f"Range {min_val}-{max_val}:")
        print("Number of values:", class_frequencies[range_]['values'])
        print("Class frequencies:", class_frequencies[range_]['class_counts'].tolist())
'''

def generate_sample(vae, latent_vecs, sample_size, device, class_labels=None, sample_strategy='boundary_box', latent_vec_filter = 'none', mean=None, std=None):

    if sample_strategy == 'boundary_box':
        random_vecs, original_labels = sample_within_boundary_box(latent_vectors=latent_vecs, num_samples=sample_size*400, device=device)
        #torch.randn(sample_size*300, 64).to(device) 
    elif sample_strategy == 'adaptive_boundary':
        random_vecs, original_labels = sample_within_boundary_box(latent_vectors=latent_vecs, num_samples=sample_size*300, device=device)
    elif sample_strategy == 'latent_clustering':
        random_vecs, original_labels = sample_within_boundary_box_in_clusters(latent_vectors=latent_vecs, device=device, 
                                                            num_samples=sample_size)
    elif sample_strategy == 'gmm':
        random_vecs, original_labels = sample_by_gmm(latent_vectors=latent_vecs, device=device, 
                                                            num_samples=sample_size)
    elif sample_strategy == 'latent_clustering_filter':
        random_vecs, original_labels = sample_within_boundary_box_in_clusters_prob(latent_vectors=latent_vecs, model=vae, num_samples=sample_size*2, prob=0.5,  
                                                                                    device=device)

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

    if latent_vec_filter == 'distance':
        # vectors and labels are generated during the selection process
        mean_distance_centroids, centroids = cluster_latent_space_kmeans(lat_vecs=latent_vecs, class_labels=class_labels,
                                                                      num_clusters=2)
        numpy_vecs, numpy_label = select_latent_vectors_by_distance(random_vecs, predicted_labels, mean_distance_centroids, centroids)
        with torch.no_grad():
            numpy_vecs = torch.tensor(numpy_vecs).to(device)
            numpy_image = vae.decoder(numpy_vecs)

    elif latent_vec_filter == 'kmeans':
        with torch.no_grad():
            for i in range(0, len(random_vecs), generated_batch_size):
                random_vecs_batch = random_vecs[i:i + generated_batch_size].to(device)
                generated_sample = vae.decoder(random_vecs_batch)
                generated_sample_batches.append(generated_sample.cpu())
            # Concatenate all decoded batches
            generated_sample = torch.cat(generated_sample_batches, dim=0)

        selected_indices = select_latent_vectors_by_kmeans(random_vecs, predicted_labels, 3, 600)
        # Access selected data points and labels using the selected indices
        numpy_image = generated_sample[selected_indices]
        numpy_label = predicted_labels[selected_indices]
        numpy_label = np.array(numpy_label.cpu().numpy())
    elif latent_vec_filter == 'probability':
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
                denormalized_data = generated_sample #* std.view(1, 6, 1).to(device) + mean.view(1, 6, 1).to(device)
                generated_sample_batches.append(denormalized_data.cpu())
            # Concatenate all decoded batches
            numpy_image = torch.cat(generated_sample_batches, dim=0)
        if sample_strategy == 'boundary_box' or sample_strategy == 'adaptive_boundary':
            numpy_label = original_labels
        elif sample_strategy == 'latent_clustering' or sample_strategy == 'gmm':
            numpy_label = original_labels
        numpy_label = np.array(numpy_label.cpu().numpy())

    # Ensure that you have the right number of samples
    print('samples.shape:', numpy_image.shape)
    numpy_image = numpy_image.squeeze().cpu().numpy()

    unique, counts = np.unique(predicted_labels.cpu().numpy(), return_counts=True)
    print(dict(zip(unique, counts)))

    unique, counts = np.unique(numpy_label, return_counts=True)
    print(dict(zip(unique, counts)))
    
    return numpy_image, numpy_label
