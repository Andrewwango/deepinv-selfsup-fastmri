from deepinv.loss.metric import PSNR, SSIM

class PSNR2(PSNR):
    def __init__(self, **kwargs):
        super().__init__(norm_inputs="min_max", complex_abs=True, **kwargs)
class SSIM2(SSIM):
    def __init__(self, **kwargs):
        super().__init__(norm_inputs="min_max", complex_abs=True, **kwargs)
class PSNR3(PSNR):
    def metric(self, x_net, x, *args, **kwargs):
        return super().metric((x_net - x_net.mean()) / x_net.std() * x.std() + x.mean(), x, *args, **kwargs)
class SSIM3(SSIM):
    def metric(self, x_net, x, *args, **kwargs):
        return super().metric((x_net - x_net.mean()) / x_net.std() * x.std() + x.mean(), x, *args, **kwargs)     