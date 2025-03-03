import math
import pickle
import random
from collections import Counter
from copy import deepcopy
from sklearn.metrics import accuracy_score
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import models.basic_model
from models import basic_model, modified_linear, model_uci
from train import exemplar_selection, architecture_update, prediction_analyzer, customized_distill_trainer
from train.visualisations import training_visualizer
from utils import data_handler
from .early_stopping_vae import *
import os
from .exemplar_size import *



OUT_PATH = 'output_reports/'
SEEDS = [i for i in range(0,30)]
BATCH_SIZE = 64
LR = 0.0005
OPTIMIZER_STEP_AFTER = 50
STEP_SIZE = 50
WEIGHT_DECAY_RATE = 1e-4

def write_to_report_and_print(outfile, string_result):
    file = open(outfile, 'a')
    file.write('\n' + string_result)
    file.close()
    print(string_result)

def new_label_map(label_map):
    new_map = {}
    for key,value in label_map.items():
        new_map[value] = key[0]
    print("new map: ", new_map)
    return new_map

def print_weights(model, layer_idx):
    idx = 0
    for layer in model.modules():
        if isinstance(layer, (nn.Conv1d, nn.Linear)):
            idx += 1
            if idx in layer_idx:
                print(f"Layer {idx} weights: {layer.weight.data}")


class Trainer():
    """
    Class for handling the training and testing processes.
    """
    def __init__(self, args):
        """
        Class initializer.
        :param args: the command line arguments
        """

        self.args = args
        print(self.args.dataset, self.args.new_classes, self.args.person)
        HOLDOUT_SIZES = calculate_exemp_size(self.args.dataset, self.args.new_classes, self.args.person)
        print('HOLDOUT_SIZES:', HOLDOUT_SIZES)
        
        self.list_of_classification_reports = {i: [] for i in HOLDOUT_SIZES}
        self.error_stats = {i: {'e_n': [], 'e_o': [], 'e_o_n': [], 'e_o_o': []} for i in HOLDOUT_SIZES}
        self.detailed_accuracies = {i: {'micro': {j: [] for j in range(1, self.args.total_classes)},
                                        'macro': {j: [] for j in range(1, self.args.total_classes)}} for i in HOLDOUT_SIZES}
        for handler, holdout_size, seed_val in self.instantiate_trainers():
            self.initialize_and_train(seed_val, handler, holdout_size=holdout_size)
            if seed_val == SEEDS[-1]:
                assert len(self.each_task_times) == len(self.all_task_times) == len(self.each_exemp_task_times) == len(self.all_exemp_task_times), \
                    "Time lengths do not match"
                cost_analyzer = f"{self.args.dataset},{self.args.new_classes}, {self.args.person}, {self.args.method}_{self.args.exemplar},{self.args.vae_lat_sampling}, {self.args.latent_vec_filter},{self.args.number}, {self.holdout_size}, {sum(self.each_task_times) / len(self.each_task_times)},{sum(self.all_task_times) / len(self.all_task_times)},{sum(self.each_exemp_task_times) / len(self.each_exemp_task_times)},{sum(self.all_exemp_task_times) / len(self.all_exemp_task_times)}" 
                write_to_report_and_print(self.outfile_t, cost_analyzer)

    def instantiate_trainers(self):
        """
        Function to instantiate the data handler class by iterating over holdout sizes and random seeds.
        """
        self.each_task_times, self.all_task_times, self.each_exemp_task_times, self.all_exemp_task_times = [], [], [], []


        HOLDOUT_SIZES = calculate_exemp_size(self.args.dataset, self.args.new_classes, self.args.person)
        print('HOLDOUT_SIZES:', HOLDOUT_SIZES)

        for holdout_size in HOLDOUT_SIZES:
            for seed in SEEDS:
                handler = self.get_dataset_handler(self.args.dataset, self.args.base_classes, self.args.new_classes, seed, self.args.person)
                yield handler, holdout_size, seed

    def initialize_and_train(self, seed, handler, holdout_size=None):
        """
        Function for training and logging.
        """
        self.initializer(seed, handler, holdout_size)
        self.train(seed, BATCH_SIZE, LR)

    def initializer(self, seed_val, handler, holdout_size=None):
        """
        Define the dataset and method-specific objects for the class.
        """
        self.seed_randomness(seed_val)
        self.holdout_size = holdout_size
        self.dataset = handler
        self.seen_cls = 0
        self.input_dim = self.dataset.getInputDim()
        self.max_size = self.holdout_size #*self.args.total_classes
        self.init_out_files()
        self.original_mapping = self.dataset.get_reversed_original_label_maps()

        self.label_map = dict(map(reversed, self.dataset.label_map.items()))
        self.visualizer = training_visualizer.Visualizer(self.args, self.dataset.num_tasks , self.holdout_size)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if any([x in self.args.method for x in ['ewc']]):
            self.model = basic_model.Net(self.input_dim, self.args.total_classes, self.args.dataset).to(self.device)
            self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
            if any([x in self.args.method for x in ['ewc']]):
                self.n_fisher_sample = None
                self.regularization_terms = {}

    def init_out_files(self):
        """
        Define the output file paths for logging
        """
        OUT_PATH = 'output_reports/'
        self.outfile, self.outfile_t = get_output_file_paths(self.args, OUT_PATH, self.holdout_size)

    @staticmethod
    def seed_randomness(seed_value):
        np.random.seed(seed_value)

    @staticmethod
    def get_dataset_handler(dataname, nb_cl, new_cl, seed, person):
        print('Person: ', person)
        handler = data_handler.DataHandler(dataname, nb_cl, new_cl, seed, person)
        return handler

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def update_model(self, itera, num_new_classes):
        """
        Method for extending the previous model with new larger number of classes
        """
        if itera > 0:
            self.previous_model = deepcopy(self.model)
            self.previous_model.to(self.device)
            self.previous_model.eval()
        lamda_mult = None
        if 'cn' in self.args.method:
            if itera == 0:
                self.model = basic_model.Net(num_new_classes, cosine_liner=True)
            else:
                if itera == 1:
                    in_features = self.model.fc.in_features
                    out_features = self.model.fc.out_features
                    new_fc = modified_linear.SplitCosineLinear(in_features, out_features,
                                                               num_new_classes)
                    new_fc.fc1.weight.data = self.model.fc.weight.data[:self.seen_cls - int(self.args.new_classes[itera-1])]  
                    old_weights = self.model.fc.weight.data[:self.seen_cls - num_new_classes].cpu().detach().numpy()
                    w_mean = np.mean(old_weights, axis=0)
                    w_std = np.std(old_weights, axis=0)
                    new_weights = np.pad(old_weights, ((0, int(self.args.new_classes[itera-1])), (0, 0)), mode="constant", constant_values=0)
                    for i in reversed(range(int(self.args.new_classes[itera-1]))):
                        for j in range(new_weights.shape[1]):
                            new_weights[new_weights.shape[0] - 1 - i][j] = np.random.normal(w_mean[j], w_std[j])
                    new_fc.fc2.weight.data = torch.nn.Parameter(torch.from_numpy(new_weights[-int(self.args.new_classes[itera-1]):]))
                    new_fc.sigma.data = self.model.fc.sigma.data[:self.seen_cls - num_new_classes]
                    self.model.fc = new_fc
                else:
                    in_features = self.model.fc.in_features
                    out_features1 = self.model.fc.fc1.out_features
                    out_features2 = self.model.fc.fc2.out_features
                    new_fc = modified_linear.SplitCosineLinear(in_features, out_features1 + out_features2,
                                                               num_new_classes)
                    new_fc.fc1.weight.data[:out_features1] = self.model.fc.fc1.weight.data
                    new_fc.fc1.weight.data[out_features1:] = self.model.fc.fc2.weight.data
                    
                    old_weights = new_fc.fc1.weight.data.cpu().detach().numpy()
                    w_mean = np.mean(old_weights, axis=0)
                    w_std = np.std(old_weights, axis=0)
                    new_weights = np.pad(old_weights, ((0, int(self.args.new_classes[itera-1])), (0, 0)), mode="constant",
                                            constant_values=0)
                    for i in reversed(range(int(self.args.new_classes[itera-1]))):
                        for j in range(new_weights.shape[1]):
                            new_weights[new_weights.shape[0] - 1 - i][j] = np.random.normal(w_mean[j], w_std[j])
                    new_fc.fc2.weight.data = torch.nn.Parameter(torch.from_numpy(new_weights[-int(self.args.new_classes[itera-1]):]))
                    new_fc.sigma.data = self.model.fc.sigma.data
                    self.model.fc = new_fc
        else:
            if itera == 0:
                if 'icarl' in self.args.exemplar:
                    self.model = basic_model.Net(num_new_classes, icarl= True, cosine_liner=False)
                else:
                    self.model = basic_model.Net(num_new_classes, icarl=False, cosine_liner=False)

                self.model.apply(lambda m: architecture_update.custom_init(m, self.model))
            else:
                self.model = architecture_update.update_model(basic_model.Net, self.model,
                                                              num_new_classes,
                                                              device=self.device)

        self.model.to(self.device)
        if itera > 0:
            old_classes_num = self.seen_cls - num_new_classes
            lamda_mult = num_new_classes * 1.0 / old_classes_num
            lamda_mult = self.args.lamda_base * math.sqrt(lamda_mult)
        return lamda_mult

    def train(self, seed, batch_size, lr):
        """
        Function to train the model over incremental batches.
        """
        all_train_class = []

        criterion = torch.nn.CrossEntropyLoss()
        all_test_data, all_test_labels = [], []
        all_val_data, all_val_labels = [], []

        test_accs = []
        task_times, exemp_times = [], []
        
        exemp = exemplar_selection.Exemplar(self.args.method, self.args.exemplar, self.args.dataset, seed, self.max_size, self.args.total_classes,
                                            self.input_dim,
                                            self.label_map,
                                            self.original_mapping, self.outfile, self.device, self.args.vae_lat_sampling, self.args.latent_vec_filter, self.args.person,
                                            self.args.base_classes, self.args.new_classes)

        for itera in range(self.dataset.num_tasks):
            to_write = ""
            train, val, test = self.dataset.getNextClasses(itera)
            train_x, train_y = zip(*train)

            if len(test) == 0:
                test_x = []
                test_y = []
                val_x = []
                val_y = []

            else:
                test_x, test_y = zip(*test)
                val_x, val_y = zip(*val)

            if len(test) == 0:
                all_test_data = all_test_data
                all_test_labels = all_test_labels
                all_val_data = all_val_data
                all_val_labels = all_val_labels
            else:
                all_test_data.extend(test_x)
                all_test_labels.extend(test_y)
                all_val_data = list(val_x)  
                all_val_labels = list(val_y)  

            print("self.dataset.classes_by_groups[itera]", self.dataset.classes_by_groups[itera])
            for n_class in self.dataset.classes_by_groups[itera]:
                if n_class not in all_train_class:
                    all_train_class.append(n_class)

            print('all_train_class:', all_train_class)

            num_new_classes = len(self.dataset.classes_by_groups[itera])
            start_time_exemp1 = timer()
            self.seen_cls = len(all_train_class)  # total num of classes seen so far
            
            self.model = self.model if hasattr(self, 'model') and itera > 0 else None

            train_xs, train_ys, all_test_data, all_test_labels, all_val_data, all_val_labels = exemplar_selection.get_exemplars_and_normalize(
                itera=itera,
                train_x=train_x,
                train_y=train_y,
                all_test_data=all_test_data,
                all_test_labels=all_test_labels,
                all_val_data=all_val_data,
                all_val_labels=all_val_labels,
                exemplar_method=self.args.exemplar,
                holdout_size=self.holdout_size,
                seen_cls=self.seen_cls,
                model=self.model,
                exemp=exemp
            )
                
            to_write += f'\nNo. of new classes: {num_new_classes}, seen classes: {self.seen_cls}, ' \
                        f'total classes: {self.args.total_classes}\n'

            train_data = torch.utils.data.DataLoader(basic_model.Dataset(train_xs, train_ys),
                                                    batch_size=batch_size,
                                                    shuffle=True, drop_last=False)
            
            end_time_exemp1 = timer() - start_time_exemp1
            test_data_all = torch.utils.data.DataLoader(basic_model.Dataset(all_test_data, all_test_labels),
                                                        batch_size=batch_size,
                                                        shuffle=False)
            val_data_all = torch.utils.data.DataLoader(basic_model.Dataset(all_val_data, all_val_labels),
                                                        batch_size=batch_size,
                                                        shuffle=False)

            to_write += f'\nIncremental batch: {itera}, Size of train set: {len(train_data.dataset)}, validation: {len(val_data_all.dataset)}, ' \
                        f'test set: {len(test_data_all.dataset)}\n'
            write_to_report_and_print(self.outfile, to_write)

            test_acc = []
            
            cur_lamda = self.update_model(itera, num_new_classes)


            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY_RATE)

            if itera > 0 and self.args.method not in ['ce', 'ce_ewc']:

                custom_trainer = customized_distill_trainer.CustomizedTrainer(self.args, itera, self.seen_cls,
                                                                              train_data, self.model,
                                                                              self.previous_model,
                                                                              cur_lamda,
                                                                              self.dataset.label_map,
                                                                              self.dataset.classes_by_groups,
                                                                              self.device)

            start_time_train = timer()

            early_stopping = EarlyStopping(patience=5, min_delta=0.1)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

            for epoch in range(self.args.epochs):
                self.model.train()

                cur_lr = self.get_lr(optimizer)
                print(f"---" * 50 + f"\nEpoch: {epoch} Current Learning Rate : {cur_lr}")

                if itera == 0:
                    print("train data shape: ", len(train_data))
                    train_loss = self.normal_training(train_data, criterion, optimizer, itera)
                elif itera > 0:
                    if self.args.method in ['ce', 'ce_ewc']:
                        train_loss = self.normal_training(train_data, criterion, optimizer, itera)

                    else:
                        print("size: ========= ", len(train_data.dataset))
                        train_loss = custom_trainer.distill_training(optimizer, num_new_classes, self.args.dataset)

                self.model.eval()
                if itera == 0:
                    val_loss, val_accuracy = self.validate_data(self.model, val_data_all)
                else:
                    val_loss, val_accuracy = self.validate_data(self.model, val_data_all)
                
                print('Train Loss: ' , train_loss, 'Val Loss:', val_loss, 'Val Accuracy:', val_accuracy) #, 'Val Loss:', val_loss, 'Val Accuracy:', val_accuracy)
                scheduler.step(val_loss)
                early_stopping.step(val_loss)
                if early_stopping.should_stop:
                    print(f"Stopping early at epoch {epoch}")
                    break

            test_stats = ""
            if any([x in self.args.method for x in ['ewc']]):
                print("Calculating importance score for parameters..")
                task_param = {}
                for n, p in self.params.items():
                    task_param[n] = p.clone().detach()
                importance = self.calculate_importance(train_data, criterion)

                if 'ewc' in self.args.method:
                    self.regularization_terms = {'importance': importance, 'task_param': task_param}

            if itera > 0 and not any([x in self.args.method for x in ['ce', 'ewc']]):
                # do weight alignment and bias correction after normal training
                self.model.train(False)
                custom_trainer.remove_hooks()

            start_time_exemp2 = timer()

            if self.args.exemplar =='vae':
                vae_trainX = np.concatenate((train_x, val_x), axis=0) 
                vae_trainy = np.concatenate((train_y, val_y)) 
                
                exemp.update(self.args.exemplar, num_new_classes, itera, train = (vae_trainX, vae_trainy), model = self.model)
            
            else:
                exemp.update(self.args.exemplar, num_new_classes, itera, train = (train_x, train_y), model = self.model)

            end_time_exemp2 = timer() - start_time_exemp2
            end_time_exemp = end_time_exemp1 + end_time_exemp2

            end_time_train = timer() - start_time_train
            task_times.append(end_time_train)

            exemp_times.append(end_time_exemp)

            self.model.eval()
            acc = self.test(test_data_all, itera, seed, exemp, final_test=True)
            test_stats += f" Final test accuracy on this task: {acc}"
            test_acc.append(acc)
            test_accs.append(max(test_acc))
            test_stats += f"\nTest accuracies over tasks {itera}: {test_accs}"
            test_stats += "\n ---------------------------------------------\n\n"
            write_to_report_and_print(self.outfile, test_stats)

        self.each_task_times.append(sum(task_times) / len(task_times))
        self.all_task_times.append(sum(task_times))
        self.each_exemp_task_times.append(sum(exemp_times) / len(exemp_times))
        self.all_exemp_task_times.append(sum(exemp_times))

        new_line = "---" * 50
        separator = "\n" + new_line
        write_to_report_and_print(self.outfile, "Random sequence of classes: {}".format([self.original_mapping[item]
                                                                                         for each in
                                                                                         self.dataset.classes_by_groups
                                                                                         for item
                                                                                         in each]) + separator * 2)

    def calculate_importance(self, dataloader, criterion):
        """
        Method to update the diagonal fisher information.
        """

        # compute a new importance matrix for the current task
        importance = {}
        for n, p in self.params.items():
            importance[n] = p.clone().detach().fill_(0)  # zero initialized

        if 'ewc' in self.args.method and self.n_fisher_sample is not None:
            # estimate fisher matrix using a subset of data; saves computation time
            n_sample = min(self.n_fisher_sample, len(dataloader.dataset))
            rand_ind = random.sample(list(range(len(dataloader.dataset))), n_sample)
            subdata = torch.utils.data.Subset(dataloader.dataset, rand_ind)
            dataloader = torch.utils.data.DataLoader(subdata, shuffle=True, batch_size=1)

        self.model.eval()
        for i, (input, target) in enumerate(dataloader):
            input, target = Variable(input), Variable(target)
            input = input.to(self.device)
            target = target.view(-1).to(self.device)
            preds = self.model(input.float())[:, :self.seen_cls]

            loss = criterion(preds, target)
            self.model.zero_grad()
            loss.backward()
            for n, p in importance.items():
                if self.params[n].grad is not None:
                    if 'ewc' in self.args.method:
                        importance[n] += ((self.params[n].grad ** 2) * len(input) / len(dataloader))
                    elif 'mas' in self.args.method:
                        importance[n] += (self.params[n].grad.abs() / len(dataloader))
        return importance

    def validate_data(self, model, val_loader, is_first_task=True):

        total_correct = 0
        total_samples = 0
        total_classification_loss = 0.0
        loss_ce = nn.CrossEntropyLoss()

        with torch.no_grad(): 
            for data, targets in val_loader:
                data  = data.float().to(self.device)
                targets = targets.long().squeeze(1).to(self.device)

                if is_first_task:
                    predictions = model(data, is_feature_input=False)
                else:
                    predictions = model(data, is_feature_input=True)

                _, predicted = predictions.max(1)
                total_correct += predicted.eq(targets).sum().item()

                classification_loss = loss_ce(predictions, targets)
                total_samples += targets.size(0)

                total_classification_loss += classification_loss.item()

        avg_ce_loss = total_classification_loss / len(val_loader)
        avg_accuracy = total_correct / total_samples

        return avg_ce_loss, avg_accuracy

    def normal_training(self, data_loader, criterion, optimizer, itera, is_first_task=True):
        """
        Method for normal training with only cross entropy
        """
        print("Training ... ")
        losses = []
        for i, (data, label) in enumerate(tqdm(data_loader)):
            data, label = Variable(data), Variable(label)

            data = data.to(self.device)
            label = label.view(-1).to(self.device)
            if is_first_task:
                p = self.model(data.float(), is_feature_input=False)
            else:
                p = self.model(data.float(), is_feature_input=True)

            loss = criterion(p, label)

            if 'ewc' in self.args.method:
                if self.regularization_terms:
                    reg_loss = 0
                    task_reg_loss = 0
                    importance = self.regularization_terms['importance']
                    task_param = self.regularization_terms['task_param']
                    for n, p in self.params.items():
                        task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()
                    reg_loss += task_reg_loss
                    loss += reg_loss * self.args.reg_coef if self.args.reg_coef > 0 else reg_loss

            optimizer.zero_grad()
            loss.backward(retain_graph=itera == 0)
            optimizer.step()
            losses.append(loss.item())
        return sum(losses) / len(data_loader)

    def test(self, testdata, itera, seed, exemp, final_test=False):
        k = 3 if itera > 0 else 2
        y_pred, y_true = [], []
        top1_acc, topk_acc = [], []

        for i, (data, label) in enumerate(testdata):
            data, label = Variable(data), Variable(label)
            data = data.to(self.device)
            label = label.view(-1).to(self.device)

            if self.args.method == 'kd_kldiv' and self.args.exemplar == ' icarl':
                with torch.no_grad():
                    p = self.model.feature_extraction(data.float())

                classes = list(exemp.mean_feature_vector.keys())
                mean_vectors = torch.stack(list(exemp.mean_feature_vector.values())).to(self.device)
                distances = torch.zeros(p.size(0), len(classes))
                for i, feature_vector in enumerate(p):
                    distances[i] = torch.norm(feature_vector - mean_vectors, p=2, dim=1)
                _, min_indices = torch.min(distances, dim=1)
                preds_list = [classes[idx] for idx in min_indices]
                labels_list = label.cpu().tolist()
            else:
                with torch.no_grad():
                    p = self.model(data.float())
                pred = p[:, :self.seen_cls].argmax(dim=-1)
                labels_list = label.cpu().tolist()
                preds_list = pred.cpu().tolist()
            prec1, preck = self.topk_accuracy(p.data, label, topk=(1, k))
            top1_acc.append(prec1.item())
            topk_acc.append(preck.item())
            y_true += [self.original_mapping[self.label_map[each]] for each in labels_list]
            y_pred += [self.original_mapping[self.label_map[each]] for each in preds_list]

        report = classification_report(y_pred=y_pred, y_true=y_true, output_dict=True)
        top1_mean = accuracy_score(y_true, y_pred)
        mispred_results = self.analyze_mispredictions(y_true, y_pred, itera)
        df = pd.DataFrame(report).transpose()
        print(df)

        result = f"Test acc: top1 = {top1_mean}, Test set size: {len(y_true)}"
        result += mispred_results
        print(result)

        if itera > 0:
            self.compute_prediction_distance_by_states(y_pred, y_true, itera, final_test=final_test)

        if final_test:
            self.compute_prediction_distance_by_classes(y_pred, y_true, itera)
            df.to_csv(self.outfile, sep='\t', mode='a')
            if itera > 0:
                detailed_acc_ = self.evaluate_detailed_accuracies(y_pred, y_true, itera)
                result += detailed_acc_
            write_to_report_and_print(self.outfile, result)
            
            if itera == self.dataset.num_tasks - 1:
                key_ = self.holdout_size
                self.list_of_classification_reports[key_].append(report)

        return top1_mean

    def topk_accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k
        Source: https://github.com/EdenBelouadah/class-incremental-learning/blob/master/il2m/codes/utils/Utils.py#L11
        """
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def compute_confusion_matrix(self, y_true, y_pred, itera):
        """
        Confusion matrix of true and predicted classes
        """
        classes_seen = [self.original_mapping[item] for each in self.dataset.classes_by_groups[:itera + 1]
                        for item in each]
        classes_seen.sort()
        conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=classes_seen)
        df_conf_mat = pd.DataFrame(conf_mat, columns=classes_seen, index=classes_seen)
        return df_conf_mat

    def analyze_mispredictions(self, y_true, y_pred, itera):
        """
        Compares the confusion between old and new classes
        """
        analyzer = prediction_analyzer.PredictionAnalysis(y_true, y_pred, self.dataset, self.original_mapping,
                                                          self.label_map)
        mispredicted_stats = analyzer.analyze_misclassified_instances(batch_num=itera)
        if itera > 0:
            misclassified_new_classes, total_new_classes, misclassified_old_classes, total_old_classes, old_misclassified_new, old_misclassified_old = mispredicted_stats
            mispred = f"\n e(n): {misclassified_new_classes}, total_new: {total_new_classes} e(o): {misclassified_old_classes}," \
                      f" total_old: {total_old_classes}, e(o,n): {old_misclassified_new}, e(o,o): {old_misclassified_old}"
            if itera == self.dataset.num_tasks - 1 and self.args.average_over != 'na':
                key_ = self.holdout_size if self.args.average_over == 'holdout' else self.train_percent
                self.error_stats[key_]['e_o'].append(misclassified_old_classes)
                self.error_stats[key_]['e_n'].append(misclassified_new_classes)
                self.error_stats[key_]['e_o_n'].append(old_misclassified_new)
                self.error_stats[key_]['e_o_o'].append(old_misclassified_old)

        else:
            misclassified_new_classes, total_new_classes = mispredicted_stats
            mispred = f"\n e(n): {misclassified_new_classes}, total_new: {total_new_classes}"
        return mispred

    def evaluate_detailed_accuracies(self, y_pred, y_true, itera):
        """
        Computes averaged accuracies over base, old and new classes after training is over
        """
        new_classes = [self.original_mapping[each] for each in self.dataset.classes_by_groups[itera]]
        old_classes = [self.original_mapping[item] for each in self.dataset.classes_by_groups[:itera]
                       for item in each]
        base_classes = [self.original_mapping[each] for each in self.dataset.classes_by_groups[0]]
        old_indices, base_indices, new_indices = [[idx for idx, val in enumerate(y_true) if val in each] for each in
                                                  [old_classes, base_classes, new_classes]]

        for indices in [base_indices, old_indices, new_indices]:
            # base, old and new classes
            filtered_list = [(pred_val, true_val) for idx, (pred_val, true_val) in enumerate(zip(y_pred, y_true)) if
                             idx in indices]
            _y_pred = [each[0] for each in filtered_list]
            _y_true = [each[1] for each in filtered_list]
            values_pred, counts_pred = np.unique(_y_true, return_counts=True)
            print('values_pred:', values_pred)
            print('counts_pred:', counts_pred)

            self.visualizer.detailed_micro_scores[itera - 1] += (f1_score(_y_true, _y_pred, average='micro') * 100,)
            self.visualizer.detailed_macro_scores[itera - 1] += (f1_score(_y_true, _y_pred, average='macro') * 100,)

        values_pred_total, counts_pred_total = np.unique(y_pred, return_counts=True)

        self.visualizer.detailed_micro_scores[itera - 1] += (f1_score(y_true, y_pred, average='micro') * 100,)  # all classes
        self.visualizer.detailed_macro_scores[itera - 1] += (f1_score(y_true, y_pred, average='macro') * 100,)

        base_class_micro, base_class_macro = self.visualizer.detailed_micro_scores[itera - 1][0], \
                                             self.visualizer.detailed_macro_scores[itera - 1][0]
        old_class_micro, old_class_macro = self.visualizer.detailed_micro_scores[itera - 1][1], \
                                           self.visualizer.detailed_macro_scores[itera - 1][1]
        new_class_micro, new_class_macro = self.visualizer.detailed_micro_scores[itera - 1][2], \
                                           self.visualizer.detailed_macro_scores[itera - 1][2]
        all_class_micro, all_class_macro = self.visualizer.detailed_micro_scores[itera - 1][3], \
                                           self.visualizer.detailed_macro_scores[itera - 1][3]

        index_ = self.holdout_size if self.args.average_over == 'holdout' else self.train_percent
        self.detailed_accuracies[index_]['micro'][itera].append(
            [base_class_micro, old_class_micro, new_class_micro, all_class_micro])
        self.detailed_accuracies[index_]['macro'][itera].append(
            [base_class_macro, old_class_macro, new_class_macro, all_class_macro])

        result = f"\n \t\t\t F1-micro \t\t F1-macro \n"
        result += f"Base classes: {base_class_micro} \t {base_class_macro}\n"
        result += f"Old classes: {old_class_micro} \t {old_class_macro}\n"
        result += f"New classes: {new_class_micro} \t {new_class_macro}\n"
        result += f"All classes: {all_class_micro} \t {all_class_macro}\n"

        return result

    @staticmethod
    def get_mean_scores(y_pred, y_true):
        """
        Computes mean scores by each class label
        """
        all_classes = set(y_true)
        class_to_scores = {_class: 0.0 for _class in all_classes}
        for each_class in all_classes:
            total_count = y_true.count(each_class)
            true_count = len([idx for idx, (first, second) in enumerate(zip(y_true, y_pred)) if
                              first == second and first == each_class])
            class_to_scores[each_class] = true_count / total_count
        return class_to_scores

    def compute_prediction_distance_by_states(self, y_pred, y_true, itera, final_test=False):
        """
        Calculated accuracy by incremental states for old and new class samples
        """
        current_classes = [self.original_mapping[each] for each in
                           self.dataset.classes_by_groups[itera]]
        previous_classes = [self.original_mapping[item] for each in
                            self.dataset.classes_by_groups[:itera] for item in each]

        class_to_scores = self.get_mean_scores(y_pred, y_true)
        previous_class_mean = sum([class_to_scores[key] for key, val in class_to_scores.items() if key in
                                   previous_classes]) / float(len(previous_classes))
        current_class_mean = sum([class_to_scores[key] for key, val in class_to_scores.items() if key in
                                  current_classes]) / float(len(current_classes))

        self.visualizer.incr_state_to_scores[itera] = (previous_class_mean, current_class_mean)

    def compute_prediction_distance_by_classes(self, y_pred, y_true, itera):
        """
        Calculated accuracy by classes when they were new and averaged over all succeeding states when they were old
        """
        current_classes = [self.original_mapping[each] for each in
                           self.dataset.classes_by_groups[itera]]
        previous_classes = [self.original_mapping[item] for each in
                            self.dataset.classes_by_groups[:itera] for item in each]
        class_to_scores = self.get_mean_scores(y_pred, y_true)
        self.visualizer.current_class_scores.update(
            {key: val for key, val in class_to_scores.items() if key in current_classes})
        for key, val in class_to_scores.items():
            if key in previous_classes:
                if key not in self.visualizer.previous_class_scores:
                    self.visualizer.previous_class_scores[key] = [val]
                else:
                    self.visualizer.previous_class_scores[key].append(val)
