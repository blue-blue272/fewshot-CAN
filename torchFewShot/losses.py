from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)

        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(inputs.size(0), inputs.size(1)).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets = targets.unsqueeze(-1)
        targets = targets.cuda()
        loss = (- targets * log_probs).mean(0).sum() 
        return loss / inputs.size(2)
