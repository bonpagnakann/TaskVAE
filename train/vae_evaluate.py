import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from .vae import *

def testing_accuracy(test_loader, model):
    correct = 0
    total = 0
    with torch.no_grad():
        for features, targets in test_loader:
            #embeddings, z_mean, z_log_var = model.encoding_fn(images, torch.nn.functional.one_hot(labels, 6))
            embeddings, z_mean, z_log_var = model.encoding_fn(features)
            outputs = model.classify(z_mean)
            _, predicted = torch.max(outputs.data, 1)
            #print('predicted:', predicted)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    print('total test size:', total )
    accuracy = 100 * correct / total
    print(f'Accuracy on classifier: {accuracy}%')

def compute_epoch_loss_autoencoder(model, data_loader, loss_fn):
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features, _ in data_loader:
            features = features
            logits = model(features)
            loss = loss_fn(logits, features, reduction='sum')
            num_examples += features.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss

def test_generated_data(generated_samples, generated_label, batch_size):
    model_save_path = './model_UCI_test_vae/model.pt'

    tested_model = Cnn1(output_dim=6)
    tested_model.load_state_dict(torch.load(model_save_path))
    tested_model.eval()


    #tested_model = load_model(model_save_path, is_gpu=False)

    #tested_model.eval()

    #tensor_labels = [target_class] * batch_size
    tensor_labels = generated_label
    #generated_samples = generated_samples.detach().numpy()
    #tensor_labels = np.array(tensor_labels)
    print('generated_samples.shape:', generated_samples.shape)
    print('tensor_labels.shape:', tensor_labels.shape)

    # Convert numpy arrays to PyTorch tensors
    tensor_samples = torch.tensor(generated_samples).float()
    tensor_labels = torch.tensor(tensor_labels).long()

    predicted_probs = tested_model(tensor_samples)
    predicted_labels = torch.argmax(predicted_probs, dim=1)

    # Convert predicted labels and ground truth labels to NumPy arrays
    predicted_labels = predicted_labels.cpu().detach().numpy()
    ground_truth_labels = np.array(tensor_labels)

    print('predicted_labels', predicted_labels)
    print('ground_truth_labels', ground_truth_labels)

    unique, counts = np.unique(predicted_labels, return_counts=True)
    print('predicted_labels:', dict(zip(unique, counts)))

    # Calculate accuracy by classes
    unique_classes = np.unique(ground_truth_labels)
    accuracy_by_class = {}

    for class_label in unique_classes:
        class_mask = (ground_truth_labels == class_label)
        correct_predictions = np.sum(predicted_labels[class_mask] == ground_truth_labels[class_mask])
        total_samples = np.sum(class_mask)

        accuracy = correct_predictions / total_samples
        accuracy_by_class[class_label] = accuracy

    print('accuracy_by_class:', accuracy_by_class)