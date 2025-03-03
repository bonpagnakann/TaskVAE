import torch
import math

def compute_margin_ranking_loss(logits, minibatch_labels, dataset, num_new_classes, seen_classes, device, outputs_bs):
    
    if dataset == 'hhar' or dataset == 'realworld' or dataset == 'uci':
        lw_mr = 1
    elif dataset == 'motion':
        lw_mr = 0.3
    elif dataset == 'pamap': 
        lw_mr = 1.5

    K, dist = 2, 0.5
    assert (outputs_bs.size() == logits.size())

    # compute ground truth scores
    high_response_index = torch.zeros(outputs_bs.size()).to(device)
    high_response_index = high_response_index.scatter(1, minibatch_labels.view(-1, 1), 1).ge(dist) 
    high_response_scores = outputs_bs.masked_select(high_response_index) 
    
    # compute top-K scores on none high response classes
    none_gt_index = torch.zeros(outputs_bs.size()).to(device)
    none_gt_index = none_gt_index.scatter(1, minibatch_labels.view(-1, 1), 1).le(dist) 
    none_gt_scores = outputs_bs.masked_select(none_gt_index).reshape((outputs_bs.size(0), logits.size(1) - 1)) 
    hard_negatives_scores = none_gt_scores.topk(K, dim=1)[0] 
    
    
    
    hard_negatives_index = minibatch_labels.lt(seen_classes - num_new_classes) 
    hard_negatives_num = torch.nonzero(hard_negatives_index).size(0) 
    

    if hard_negatives_num > 0:
        gt_scores = high_response_scores[hard_negatives_index].view(-1, 1).repeat(1, K) 
        hard_scores = hard_negatives_scores[hard_negatives_index] 
        
        assert (gt_scores.size() == hard_scores.size())
        assert (gt_scores.size(0) == hard_negatives_num)
        mr_loss = torch.nn.MarginRankingLoss(margin=dist)(gt_scores.view(-1, 1), hard_scores.view(-1, 1),
                                                          torch.ones(hard_negatives_num * K).view(-1, 1).to(device)) * lw_mr
    else:
        mr_loss = torch.tensor(0.).to(device)

    return mr_loss