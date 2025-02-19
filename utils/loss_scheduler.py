from typing import List
from torch import Generator, randint, Tensor, tensor
from torch.nn import Module
from deepinv.loss.loss import Loss
from deepinv.physics.forward import Physics


class BaseLossScheduler(Loss):
    r"""
    Base class for loss schedulers.

    Wraps a list of losses, and each time forward is called, some of them are selected based on a defined schedule.

    :param Loss \*loss: loss or multiple losses to be scheduled.
    :param Generator generator: torch random number generator, defaults to None
    """

    def __init__(self, *loss: Loss, generator: Generator = None):
        super().__init__()
        self.losses = loss
        self.rng = generator if generator is not None else Generator()

    def schedule(self, epoch: int) -> List[Loss]:
        r"""
        Return selected losses based on defined schedule, optionally based on current epoch.

        :param int epoch: current epoch number
        :return list[Loss]: selected (sub)list of losses to be used this time.
        """
        return self.losses

    def forward(
        self,
        x_net: Tensor = None,
        x: Tensor = None,
        y: Tensor = None,
        physics: Physics = None,
        model: Module = None,
        epoch: int = None,
        **kwargs,
    ):
        r"""
        Loss forward pass.

        When called, subselect losses based on defined schedule to be used at this pass, and apply to inputs.

        :param torch.Tensor x_net: model output
        :param torch.Tensor x: ground truth
        :param torch.Tensor y: measurement
        :param Physics physics: measurement operator
        :param torch.nn.Module model: reconstruction model
        :param int epoch: current epoch
        """
        losses = self.schedule(epoch)

        if len(losses) == 1 and isinstance(losses[0], (list, tuple)):
            losses = losses[0]

        loss_total = 0.0
        for l in losses:
            loss_total += l.forward(
                x_net=x_net,
                x=x,
                y=y,
                physics=physics,
                model=model,
                epoch=epoch,
                **kwargs,
            )
        if isinstance(loss_total, float):
            return tensor(loss_total, requires_grad=True)
        return loss_total

    def adapt_model(self, model: Module, **kwargs):
        r"""
        Adapt model using all wrapped losses.

        Some loss functions require the model forward call to be adapted before the forward pass.

        :param torch.nn.Module model: reconstruction model
        """
        for l in self.losses:
            for _l in l if isinstance(l, (list, tuple)) else [l]:
                model = _l.adapt_model(model, **kwargs)
        return model


class RandomLossScheduler(BaseLossScheduler):
    r"""
    Schedule losses at random.

    The scheduler wraps a list of losses. Each time this is called, one loss is selected at random and used for the forward pass.

    Optionally pass a weighting for each loss e.g. if `weightings=[3, 1]` then the first loss is 3 times more likely to be called than the second loss.

    :Example:

    >>> import torch
    >>> from deepinv.loss import RandomLossScheduler, SupLoss
    >>> from deepinv.loss.metric import SSIM
    >>> l = RandomLossScheduler(SupLoss(), SSIM(train_loss=True)) # Choose randomly between Sup and SSIM
    >>> x_net = x = torch.tensor([0., 0., 0.])
    >>> l(x=x, x_net=x_net)
    tensor(0.)

    :param Loss \*loss: loss or multiple losses to be scheduled.
    :param Generator generator: torch random number generator, defaults to None
    """

    def __init__(self, *loss, generator=None, weightings=None):
        super().__init__(*loss, generator=generator)
        self.weightings = weightings

        if weightings is not None:
            assert len(self.losses) == len(weightings)
            self.weightings = sum([[i] * w for (i, w) in enumerate(weightings)], [])

    def schedule(self, epoch) -> List[Loss]:
        if self.weightings is None:
            choice = randint(len(self.losses), (1,), generator=self.rng).item()
        else:
            choice = self.weightings[
                randint(len(self.weightings), (1,), generator=self.rng).item()
            ]
        return [self.losses[choice]]
