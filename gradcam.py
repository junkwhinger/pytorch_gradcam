import torch
import torch.nn as nn
from torch.nn import functional as F

from torchvision import models as vmodels
from torchvision import transforms as transforms


class _PropagationBase(object):
    def __init__(self, model, target_layer):
        super(_PropagationBase, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model.eval()
        self.image = None
        self.target_layer = target_layer

    def _encode_one_hot(self, idx_list):
        one_hot = torch.zeros(self.preds.size())
        for i, idx in enumerate(idx_list):
            one_hot[i, idx] = 1.0
        return one_hot.to(self.device)

    def forward(self, image):
        self.model.zero_grad()
        self.preds = self.model(image)
        self.probs = F.softmax(self.preds, dim=1)
        return self.probs

    def backward(self, idx):
        one_hot = self._encode_one_hot(idx)
        self.preds.backward(gradient=one_hot, retain_graph=True)
        
class GradCAM(_PropagationBase):
    def __init__(self, model, target_layer):
        super(GradCAM, self).__init__(model, target_layer)
        self.fmap = None
        self.grad = None
        self.is_valid = False

        def func_f(module, input, output):
            self.fmap = output.detach()

        def func_b(module, grad_in, grad_out):
            self.grad = grad_out[0].detach()

        for module in self.model.named_modules():
            if module[0] == self.target_layer:
                self.is_valid = True
                module[1].register_forward_hook(func_f)
                module[1].register_backward_hook(func_b)

        if not self.is_valid:
            raise ValueError('Invalid layer name: {}'.format(target_layer))


    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        return F.adaptive_avg_pool2d(grads, 1)

    def generate(self):
        
        normalized_grad = F.normalize(self.grad, p=2, dim=1, eps=1e-12)
        weight = F.adaptive_avg_pool2d(normalized_grad, 1)
        
        weighted_combination = (self.fmap * weight).sum(dim=1)
        relued = torch.clamp(weighted_combination, min=0.0)
        
        min_values = relued.min(2, keepdim=True)[0].min(1, keepdim=True)[0]
        max_values = relued.max(2, keepdim=True)[0].max(1, keepdim=True)[0]
        gcam = ((relued - min_values) / max_values).cpu().numpy()
        
        
        return gcam