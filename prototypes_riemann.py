#
# Obtain hyperspherical prototypes prior to network training with Riemannian optimization.
#
# @inproceedings{mettes2016hyperspherical,
#  title={Hyperspherical Prototype Networks},
#  author={Mettes, Pascal and van der Pol, Elise and Snoek, Cees G M},
#  booktitle={Advances in Neural Information Processing Systems},
#  year={2019}
# }
#
import argparse
import os
import random
import time

import ledoh_torch
import geoopt
import numpy as np
import torch
import torch.nn.functional as F
import json

def prototype_loss(prototypes):
    # Dot product of normalized prototypes is cosine similarity.
    product = torch.matmul(prototypes, prototypes.t()) + 1
    # Remove diagnonal from loss.
    product -= 2. * torch.diag(torch.diag(product))
    # Minimize maximum cosine similarity.
    loss = product.max(dim=1)[0]
    return loss.mean()

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperspherical prototypes")
    parser.add_argument('-c', dest="classes", default=100, type=int)
    parser.add_argument('-d', dest="dims", default=100, type=int)
    parser.add_argument('-l', dest="learning_rate", default=0.01, type=float)
    parser.add_argument('-m', dest="momentum", default=0.9, type=float)
    parser.add_argument('-e', dest="epochs", default=10000, type=int)
    parser.add_argument('-s', dest="seed", default=300, type=int)
    parser.add_argument('-r', dest="resdir", default="", type=str)
    parser.add_argument('-n', dest="nn", default=2, type=int)
    parser.add_argument('-o', dest="optim", default="rsgd", type=str)
    parser.add_argument('--disp', dest="dispersion", default=None, type=str)
    parser.add_argument('--dis-params', dest="dispersion_params", default=None, type=str)
    args = parser.parse_args()
    return args

#
# Main entry point of the script.
#
if __name__ == "__main__":
    # Parse user arguments.
    args = parse_args()
    device = torch.device("cuda")

    # Set seed.
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Initialize prototypes.
    prototypes = torch.randn(args.classes, args.dims, device=device)
    prototypes = geoopt.ManifoldParameter(F.normalize(prototypes, p=2, dim=1), manifold=geoopt.SphereExact())
    if args.optim == "rsgd":
        optimizer = geoopt.optim.RiemannianSGD([prototypes], lr=args.learning_rate, momentum=args.momentum, stabilize=1)
    elif args.optim == "radam":
        optimizer = geoopt.optim.RiemannianAdam([prototypes], lr=args.learning_rate, stabilize=1)
    else:
        raise ValueError("Unknown optimizer")

    dispersion_params = json.loads(args.dispersion_params)
    disp_funcs = {
    "mmd": ledoh_torch.kernel_dispersion.KernelSphereDispersion(**dispersion_params),
    "lloyd": ledoh_torch.lloyd_dispersion.LloydSphereDispersion(**dispersion_params),
    "sliced": ledoh_torch.sliced_dispersion.SlicedSphereDispersion(**dispersion_params),
    "sliced-ax": ledoh_torch.sliced_batch.AxisAlignedBatchSphereDispersion(**dispersion_params),
    "hpn": prototype_loss
    }
    dispersion_fn = disp_funcs[args.dispersion]

    for i in range(args.epochs):
        optimizer.zero_grad()
        loss = dispersion_fn(prototypes)
        loss.backward()
        optimizer.step()

    d_min = ledoh_torch.minimum_acos_distance(prototypes.detach().clone(), prototypes.detach().clone())
    c_var = ledoh_torch.circular_variance(prototypes.detach().clone())
    print(f"dim: {args.dims} classes: {args.classes} dispersion: {args.dispersion} lr: {args.learning_rate} epochs: {args.epochs}")
    print(f"Dmin {d_min.item()} Cvar {c_var.item()}")

    # Store result.
    os.makedirs(args.resdir, exist_ok=True)
    if args.dispersion is not None:
        disp_params_save = args.dispersion_params.replace('"', '').replace("{", "").replace("}", "").replace(":", "-").replace(",", "-").replace(" ", "")
        fn = f"{args.resdir}/prototypes-{args.dims}d-{args.classes}c-{args.dispersion}-{disp_params_save}-{args.learning_rate}lr.npy"
    else:
        fn = f"{args.resdir}/prototypes-{args.dims}d-{args.classes}c.npy"
    np.save(fn, prototypes.cpu().data.numpy())
