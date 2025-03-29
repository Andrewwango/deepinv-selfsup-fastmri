from argparse import ArgumentParser

import numpy as np
from scipy.sparse.linalg import svds
import torch

import deepinv as dinv
from deepinv.datasets.fastmri import FastMRISliceDataset, FullMultiCoilFastMRITransform
from deepinv.physics.mri import MultiCoilMRI, MRIMixin

class ScipyOp(MRIMixin):
    def __init__(self, physics, n_coils=16, img_size=(384, 320)):
        self.physics = physics
        self.img_size = img_size
        self.N, H, W = n_coils, *img_size
        self.shape = (H * W * n_coils, H * W)
        self.dtype = np.complex64

    def np2torch(self, x, shape):
        return self.from_torch_complex(torch.tensor(x.reshape(shape), dtype=torch.complex64).unsqueeze(0))

    def torch2np(self, x):
        return self.to_torch_complex(x).squeeze(0).flatten().numpy()

    def matvec(self, v):
        v = self.np2torch(v, shape=self.img_size) # #1,2,H,W
        return self.torch2np(self.physics.A(v)) # (H*W*N)

    def rmatvec(self, u):
        u = self.np2torch(u, shape=(self.N, *self.img_size)) # #1,2,N,H,W
        return self.torch2np(self.physics.A_adjoint(u)) # (H*W)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--acc", type=int, default=2)
    parser.add_argument("--n_coils", type=int, default=4)
    parser.add_argument("--img_size", type=tuple, default=(180, 180))
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--eps", type=float, default=1e-3)
    args = parser.parse_args()

    rng = torch.Generator().manual_seed(0)
    mask_generator = dinv.physics.generator.RandomMaskGenerator(img_size=args.img_size, acceleration=args.acc, rng=rng)

    dataset = FastMRISliceDataset(
        "../fastmri-experiments/local_data/volumes/brain",
        slice_index="middle",
        transform=FullMultiCoilFastMRITransform(
            mask_generator=mask_generator,
            crop_size=args.img_size,
        ),
        filter_id=lambda s: s.metadata["coils"] == args.n_coils,
    )

    iterator = iter(torch.utils.data.DataLoader(dataset))
    x, y, params = next(iterator)
    physics = MultiCoilMRI(**params)

    svd = svds(
        ScipyOp(physics, args.n_coils, args.img_size),
        k=args.k,
        return_singular_vectors=False,
        solver="lobpcg",
        which="LM", #"SM"
    )

    np.save(f"svd_{args.acc}_{args.n_coils}_{args.img_size[0]}_{args.img_size[1]}.npy", svd)

    effective_rank = np.sum(svd > args.eps)
    print(f"Effective rank: {effective_rank}")

# python scripts/estimate_svd.py --k 50