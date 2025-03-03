
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from .vae_evaluate import *
from .early_stopping_vae import *
import torch.optim as optim


def validate(model, val_loader, device):
    model.eval()  # Set the model to evaluation mode

    total_correct = 0
    total_samples = 0
    total_reconstruction_loss = 0.0
    total_classification_loss = 0.0
    loss_mse = F.mse_loss
    loss_ce = nn.CrossEntropyLoss()

    with torch.no_grad():  # Disable gradient computation during evaluation
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)

            reconstructed_data, z_mean, z_log_var, predictions = model(data)

            reconstruction_loss = loss_mse(reconstructed_data, data, reduction='none')
            reconstruction_loss = reconstruction_loss.view(data.size(0), -1).sum(axis=1)  # sum over pixels
            reconstruction_loss = reconstruction_loss.mean()  # average over batch dimension

            _, predicted = predictions.max(1)
            total_correct += predicted.eq(targets).sum().item()

            classification_loss = loss_ce(predictions, targets)
            total_samples += targets.size(0)

            total_reconstruction_loss += reconstruction_loss.item()
            total_classification_loss += classification_loss.item()

    avg_ce_loss = total_classification_loss / len(val_loader)
    avg_reconstruction_loss = total_reconstruction_loss / len(val_loader)
    avg_accuracy = total_correct / total_samples

    return avg_ce_loss, avg_accuracy, avg_reconstruction_loss


def train_vae(num_epochs, model, optimizer,
                 train_loader, val_loader, device, loss_fn=None,
                 kl_coefficient=0.0001,
                 ce_coefficient=65,
                 save_model=None):
    log_dict = {'train_combined_loss_per_epoch': [],
                'train_ce_acc_per_epoch': [],
                'train_reconstruction_loss_per_epoch': [],
                'train_ce_loss_per_epoch': [],
                'train_ce_val_loss_per_epoch': [],
                'train_ce_val_acc_per_epoch': [],
                'train_kl_loss_per_epoch': [],
                'train_val_recon_loss_per_epoch': []}

    if loss_fn is None:
        loss_fn = F.mse_loss
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    early_stopping = EarlyStopping(patience=7, min_delta=5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    stopped_epochs = []

    best_val_loss = float('inf')
    no_improve_epochs = 0
    stop_classifier_training = False
    patience = 5

    for epoch in range(num_epochs):

        model.train()
        total_correct = 0
        total_reconstruction_loss, total_kl_loss, total_classification_loss, total_combined_loss = 0.0, 0.0, 0.0, 0.0


        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)

            # FORWARD AND BACK PROP
            decoded, z_mean, z_log_var, predictions = model(features)

            kl_loss = -0.5 * torch.sum(1 + z_log_var
                                      - z_mean ** 2
                                      - torch.exp(z_log_var),
                                      axis=1)  # sum over latent dimension

            batchsize = kl_loss.size(0)
            kl_loss = kl_loss.mean()  # average over batch dimension

            reconstruction_loss = loss_fn(decoded, features, reduction='none')
            reconstruction_loss = reconstruction_loss.view(batchsize, -1).sum(axis=1)  # sum over pixels
            reconstruction_loss = reconstruction_loss.mean()  # average over batch dimension

            _, predicted = predictions.max(1)
            total_correct += predicted.eq(labels).sum().item()

            classification_loss = criterion(predictions, labels)

            loss =  reconstruction_loss + kl_coefficient * kl_loss 
            if not stop_classifier_training:    
                loss = loss + (classification_loss*ce_coefficient)
            
            optimizer.zero_grad()

            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            total_reconstruction_loss += reconstruction_loss.item()
            total_kl_loss += kl_loss.item()
            total_classification_loss += classification_loss.item()
            total_combined_loss += loss.item()

        # Validation
        model.eval()
        val_loss, val_accuracy, val_reconstruction_loss = validate(model, val_loader, device= device)
        avg_reconstruction_loss = total_reconstruction_loss / len(train_loader)
        avg_kl_loss = total_kl_loss / len(train_loader)
        avg_classification_loss = total_classification_loss / len(train_loader)
        avg_combined_loss = total_combined_loss / len(train_loader)
        avg_accuracy = total_correct / len(train_loader.dataset)
        print('Epoch: %03d/%03d | Batch %04d/%04d |  Loss: %.4f '
              % (
                epoch + 1, num_epochs, batch_idx+1,
                 len(train_loader), avg_combined_loss))
        print('avg_reconstruction_loss:', avg_reconstruction_loss)
        print('avg_kl_loss:', avg_kl_loss)
        print('avg_classification_loss:', avg_classification_loss)
        print('avg_combined_loss:', avg_combined_loss)
        print('avg_accuracy:', avg_accuracy)
        print('val_loss:', val_loss)
        print('val_accuracy:', val_accuracy)
        print('val_reconstruction_loss:', val_reconstruction_loss)

        print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))

        #LOGGING
        log_dict['train_combined_loss_per_epoch'].append(avg_combined_loss)
        log_dict['train_reconstruction_loss_per_epoch'].append(avg_reconstruction_loss)
        log_dict['train_ce_loss_per_epoch'].append(avg_classification_loss)
        log_dict['train_kl_loss_per_epoch'].append(avg_kl_loss)
        log_dict['train_ce_acc_per_epoch'].append(avg_accuracy)
        log_dict['train_ce_val_loss_per_epoch'].append(val_loss)
        log_dict['train_ce_val_acc_per_epoch'].append(val_accuracy)
        log_dict['train_val_recon_loss_per_epoch'].append(val_reconstruction_loss)

        if not stop_classifier_training:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    print(f"Stopping training for classifier at epoch {epoch+1}")
                    stop_classifier_training = True

        scheduler.step(val_reconstruction_loss)

        # Print current learning rate
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            print(f'Epoch {epoch}: Current learning rate = {current_lr}')

        early_stopping.step(val_reconstruction_loss)
        if early_stopping.should_stop:
            print(f"Stopping early at epoch {epoch}")
            stopped_epochs.append(epoch+1)
            break

    print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))
    print('stopped epochs:', stopped_epochs)
    if save_model is not None:
        torch.save(model.state_dict(), save_model)

    return model, log_dict