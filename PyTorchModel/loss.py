import torch
import torch.nn as nn

class OutlineConnectivityLoss(nn.Module):
    def __init__(self, smooth=1, connectivity_weight=0.1):
        super(OutlineConnectivityLoss, self).__init__()
        self.smooth = smooth
        self.connectivity_weight = connectivity_weight

    def dice_loss(self, inputs, targets):
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        return 1 - (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)

    def connectivity_penalty(self, inputs, targets):
        # Penalty on overlapping regions
        grad_h_overlap = torch.abs((inputs * targets)[:, :, 1:, :] - (inputs * targets)[:, :, :-1, :])
        grad_w_overlap = torch.abs((inputs * targets)[:, :, :, 1:] - (inputs * targets)[:, :, :, :-1])

        # Penalty on non-overlapping predicted regions (false positives)
        non_overlap = inputs * (1 - targets)  # Predicted regions that don't overlap with ground truth
        grad_h_non_overlap = torch.abs(non_overlap[:, :, 1:, :] - non_overlap[:, :, :-1, :])
        grad_w_non_overlap = torch.abs(non_overlap[:, :, :, 1:] - non_overlap[:, :, :, :-1])

        # Combine both penalties
        penalty_overlap = (grad_h_overlap.sum() + grad_w_overlap.sum()) / inputs.numel()
        penalty_non_overlap = (grad_h_non_overlap.sum() + grad_w_non_overlap.sum()) / inputs.numel()
        return penalty_overlap + penalty_non_overlap

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        dice = self.dice_loss(inputs, targets)
        penalty = self.connectivity_penalty(inputs, targets)
        return dice + self.connectivity_weight * penalty