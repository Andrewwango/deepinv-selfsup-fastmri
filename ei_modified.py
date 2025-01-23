from deepinv.loss.ei import EILoss

class ModifiedEILoss(EILoss):
    def __init__(self, *args, weight2: float = 1., **kwargs):
        super().__init__(*args, **kwargs)
        self.weight2 = weight2

    def forward(self, x_net, physics, model, **kwargs):
        transform_params = self.T.get_params(x_net)
        
        x2 = self.T(x_net, **transform_params)

        if self.noise:
            y = physics(x2)
        else:
            y = physics.A(x2)

        x3 = model(y, physics)

        x4 = self.T.inverse(x3, **transform_params)

        loss_ei = self.weight * self.metric(x3, x2) + self.weight2 * self.metric(x4, x_net)
        return loss_ei
