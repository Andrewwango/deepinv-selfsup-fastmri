from deepinv.training.adversarial import AdversarialTrainer as Temp
import torch

def n(x):
    return torch.isnan(x).any().item() or torch.isinf(x).any().item()

class AdversarialTrainer(Temp):
    def compute_loss(self, physics, x, y, train=True, epoch = None):

        if n(x) or n(y):
            print(f"NANS DETECTED COMPUTE_LOSS {n(x)} {n(y)}")

        out = super().compute_loss(physics, x, y, train, epoch)

        if n(out):
            print(f"NANS DETECTED COMPUTE_LOSS {n(out)}")

        return out
    
    def compute_metrics(self, x, x_net, y, physics, logs, train=True, epoch = None):
        if n(x) or n(y) or n(x_net):
            print(f"NANS DETECTED COMPUTE_METRICS {n(x)} {n(y)} {n(x_net)}")

        out = super().compute_metrics(x, x_net, y, physics, logs, train, epoch)

        if n(out):
            print(f"NANS DETECTED COMPUTE_METRICS {n(out)}")

        return out

    def model_inference(self, y, physics, x=None, train=True, **kwargs):
        if n(y):
            print(f"NANS DETECTED MODEL_INFERENCE {n(y)}")

        out = super().model_inference(y, physics, x, train, **kwargs)
        
        if n(out):
            print(f"NANS DETECTED MODEL_INFERENCE {n(out)}")

        return out