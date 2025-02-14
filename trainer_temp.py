from deepinv.training import Trainer
from deepinv.loss.measplit import SplittingLoss
import torch

class TempTrainer(Trainer):
    def load_model(self, ckpt_pretrained = None):
        if ckpt_pretrained is None and self.ckpt_pretrained is not None:
            ckpt_pretrained = self.ckpt_pretrained

        if ckpt_pretrained is not None:
            checkpoint = torch.load(
                ckpt_pretrained, map_location=self.device, weights_only=True
            )
            try:
                self.model.load_state_dict(checkpoint["state_dict"])
            except RuntimeError as e:
                if isinstance(self.model, SplittingLoss.SplittingModel):
                    self.model.model.load_state_dict(checkpoint["state_dict"])
                else:
                    raise e
            if "optimizer" in checkpoint and self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            if "wandb_id" in checkpoint and self.wandb_vis:
                self.wandb_setup["id"] = checkpoint["wandb_id"]
                self.wandb_setup["resume"] = "allow"
            if "epoch" in checkpoint:
                self.epoch_start = checkpoint["epoch"]