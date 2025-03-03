import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cosine
from torch.autograd import Variable
from tqdm import tqdm

from .losses import margin_ranking_loss

cur_features = []
ref_features = []
old_scores = []
new_scores = []


def get_ref_features(self, inputs, outputs):
    global ref_features
    ref_features = inputs[0]


def get_cur_features(self, inputs, outputs):
    global cur_features
    cur_features = inputs[0]

def get_old_scores_before_scale(self, inputs, outputs):
    global old_scores
    old_scores = outputs


def get_new_scores_before_scale(self, inputs, outputs):
    global new_scores
    new_scores = outputs

loss_by_epoch = dict()

class CustomizedTrainer():
    def __init__(self, args, itera, seen_cls, train_loader, model, previous_model, lamda,
                 virtual_map, classes_by_groups, device):
        self.args = args
        self.itera = itera
        self.seen_class = seen_cls
        self.train_loader = train_loader
        self.device = device
        self.cur_lamda = lamda
        self.model, self.previous_model = model, previous_model
        self.virtual_map = virtual_map
        self.handle_ref_features = self.previous_model.fc.register_forward_hook(get_ref_features)
        self.handle_cur_features = self.model.fc.register_forward_hook(get_cur_features)
        if 'cn' in self.args.method:
            self.handle_old_scores_bs = self.model.fc.fc1.register_forward_hook(get_old_scores_before_scale)
            self.handle_new_scores_bs = self.model.fc.fc2.register_forward_hook(get_new_scores_before_scale)

    def distill_training(self, optimizer, num_new_classes, dataset):
        print("Training with distillation losses... ")
        losses = []
        dataloader = self.train_loader

        for i, (feature, label) in enumerate(tqdm(dataloader)):
            feature, label = Variable(feature), Variable(label)
            feature = feature.to(self.device)
            
            if 'kd' in self.args.method:
                label = label.view(-1).to(self.device)
                label = F.one_hot(label, self.seen_class)
            else:
                label = label.view(-1).to(self.device)

            optimizer.zero_grad()
            p = self.model(feature.float())

            if 'kd' in self.args.method:
                p_old = F.sigmoid(p[:, :self.seen_class - num_new_classes])
                p_new = F.sigmoid(p[:, self.seen_class - num_new_classes:])
                label_new = label[:,self.seen_class - num_new_classes:].float()

                
            else:
                p_old = F.log_softmax(p[:, :self.seen_class - num_new_classes] / self.args.T, dim=1)
            self.previous_model.eval()
            with torch.no_grad():
                pre_p = self.previous_model(feature.float())
                assert pre_p.shape[1] == self.seen_class - num_new_classes, print("Shape mismatch between previous "
                                                                                  "model and no. of old classes.")
                pre_p = F.sigmoid(pre_p)

            if 'kd' in self.args.method:
                loss_hard_target = F.binary_cross_entropy_with_logits(p_new, label_new)
            else:
                loss_hard_target = torch.nn.CrossEntropyLoss()(p, label)
   
            if any([x in self.args.method for x in ['ce', 'cn']]):
                loss = loss_hard_target
                loss_stats = f"CE loss: {loss_hard_target}"
            elif 'kd' in self.args.method:
                # preserve previous knowledge by encouraging current predictions on old classes to match soft labels of previous model
                loss_soft_target = F.binary_cross_entropy_with_logits(p_old, pre_p) * self.args.lamda_old
                loss = 2*loss_hard_target + loss_soft_target  
                loss_stats = f"\nCE loss: {loss_hard_target}, KD loss: {loss_soft_target}"
            else:
                print("Invalid distillation method (kd or cn)")
            if 'lfc' in self.args.method:
                # less forget constraint loss
                cur_features_ = F.normalize(cur_features, p=2, dim=1)
                ref_features_ = F.normalize(ref_features.detach(), p=2, dim=1)
                less_forget_constraint = torch.nn.CosineEmbeddingLoss()(cur_features_, ref_features_,
                                                                        torch.ones(feature.shape[0]).to(
                                                                            self.device)) * self.cur_lamda
                loss += less_forget_constraint
                loss_stats += f" LFC loss: {less_forget_constraint}"
            if 'mr' in self.args.method:
                # compute margin ranking loss
                if 'cn' in self.args.method:
                    output_bs = torch.cat((old_scores, new_scores), dim=1)
                else:
                    output_bs = p
                    output_bs = F.normalize(output_bs, p=2, dim=1)
                mr_loss = margin_ranking_loss.compute_margin_ranking_loss(p, label, dataset,num_new_classes, self.seen_class,
                                                                          self.device, output_bs)
                loss += mr_loss
                loss_stats += f" MR loss: {mr_loss}"

            print(loss_stats)
            loss.backward(retain_graph=True)
            optimizer.step()

            losses.append(loss.item())

        return sum(losses) / len(dataloader.dataset)

    def remove_hooks(self):
        # remove the registered hook after model has been trained for the incremental batch
        self.handle_ref_features.remove()
        self.handle_cur_features.remove()
        if 'cn' in self.args.method:
            self.handle_old_scores_bs.remove()
            self.handle_new_scores_bs.remove()
