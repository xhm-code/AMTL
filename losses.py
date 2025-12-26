import torch
import torch.nn.functional as F
from copy import deepcopy

def get_label(args, logits, num_labels=-1): #Balance_Softmax
    if args.loss_type == 'balance_softmax'or args.loss_type=='Balance_Softmax' or args.loss_type=='Balance_Softmax':
        th_logit = torch.zeros_like(logits[..., :1])
    else:
        th_logit = logits[:, 0].unsqueeze(1)
    output = torch.zeros_like(logits).to(logits)
    mask = (logits > th_logit)
    if num_labels > 0:
        top_v, _ = torch.topk(logits, num_labels, dim=1)
        top_v = top_v[:, -1]
        mask = (logits >= top_v.unsqueeze(1)) & mask
    output[mask] = 1.0
    output[:, 0] = (output.sum(1) == 0.).to(logits)
    return output

def get_label_by_MT(args, logits, num_labels=-1, Na_idxs=[(0, 11), (11,32),(32,79), (79,100)]):
    Na_start_idxs = [start for start, _ in Na_idxs]
    outputs = []
    for (start, end) in Na_idxs:
        new_logits = logits[:, start:end]
        if args.loss_type == 'balance_softmax'or args.loss_type=='Balance_Softmax' or args.loss_type=='Balance_Softmax_MT':
            th_logit = torch.zeros_like(new_logits[..., :1])
        else:
            th_logit = new_logits[:, 0].unsqueeze(1)

        output = torch.zeros_like(new_logits)
        mask = new_logits > th_logit

        if num_labels > 0:
            top_v, _ = torch.topk(new_logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (new_logits >= top_v.unsqueeze(1)) & mask

        output[mask] = 1.0
        if start in Na_start_idxs:
            output[output.sum(1) == 0, 0] = 0
        outputs.append(output)
    outputs = torch.cat(outputs, dim=-1)
    outputs[:, 0] = (outputs.sum(1) == 0)

    return outputs

def get_at_loss(logits, labels,):
    """
    ATL
    """
    labels = deepcopy(labels)
    # TH label
    th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
    th_label[:, 0] = 1.0
    labels[:, 0] = 0.0
    p_mask = labels + th_label
    n_mask = 1 - labels
    # Rank positive classes to TH
    logit1 = logits - (1 - p_mask) * 1e30
    loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)
    # Rank TH to negative classes
    logit2 = logits - (1 - n_mask) * 1e30
    loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)
    # Sum two parts
    loss = loss1 + loss2
    loss = loss.mean()
    return loss

def get_at_loss_by_MT(logits, labels, Na_idxs=[(0, 11), (11, 32), (32, 79), (79, 100)], MT_lambda=3.5):
    losses = []

    Na_start_idxs = [start for start, _ in Na_idxs]
    for (start, end) in Na_idxs:

        new_logits = logits[:, start:end].clone()
        new_labels = labels[:, start:end].clone()

        th_label = torch.zeros_like(new_labels, dtype=torch.float).to(labels.device)
        th_label[:, 0] = 1.0
        new_labels[:, 0] = 0.0

        other_idxs = [idx for idx in Na_start_idxs if idx != start]
        selected_logits_sum = torch.sum(logits[:, other_idxs], dim=1)
        new_logits[:, 0] = (new_logits[:, 0] + selected_logits_sum) / MT_lambda
        p_mask = new_labels + th_label
        n_mask = 1 - new_labels

        logit1 = F.log_softmax(new_logits - (1 - p_mask) * 1e30, dim=-1)
        loss1 = -(logit1 * new_labels).sum(1)

        logit2 = F.log_softmax(new_logits - (1 - n_mask) * 1e30, dim=-1)
        loss2 = -(logit2 * th_label).sum(1)

        loss = loss1 + loss2
        losses.append(loss)

    return torch.stack(losses, dim=0).mean()