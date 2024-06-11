#
# Obtain hyperspherical prototypes prior to network training.
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
import sys
import time

import ledoh_torch
import geoopt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn


#
# PArse user arguments.
#
def parse_args():
    parser = argparse.ArgumentParser(description="Hyperspherical prototypes")
    parser.add_argument('-c', dest="classes", default=100, type=int)
    parser.add_argument('-d', dest="dims", default=100, type=int)
    parser.add_argument('-l', dest="learning_rate", default=0.01, type=float)
    parser.add_argument('-m', dest="momentum", default=0.9, type=float)
    parser.add_argument('-e', dest="epochs", default=10000, type=int)
    parser.add_argument('-s', dest="seed", default=300, type=int)
    parser.add_argument('-r', dest="resdir", default="", type=str)
    parser.add_argument('-w', dest="wtvfile", default="", type=str)
    parser.add_argument('-n', dest="nn", default=2, type=int)
    parser.add_argument('-o', dest="optim", default="rsgd", type=str)
    parser.add_argument('--disp', dest="dispersion", default=None, type=str)
    parser.add_argument('--dispw', dest="dispersion_weight", default=0, type=float)
    args = parser.parse_args()
    return args


#
# Compute the loss related to the prototypes.
#
def prototype_loss(prototypes):
    # Dot product of normalized prototypes is cosine similarity.
    product = torch.matmul(prototypes, prototypes.t()) + 1
    # Remove diagnonal from loss.
    product -= 2. * torch.diag(torch.diag(product))
    # Minimize maximum cosine similarity.
    loss = product.max(dim=1)[0]
    return loss.mean(), product.max()


#
# Compute the semantic relation loss.
#
def prototype_loss_sem(prototypes, triplets):
    product = torch.matmul(prototypes, prototypes.t()) + 1
    product -= 2. * torch.diag(torch.diag(product))
    loss1 = -product[triplets[:, 0], triplets[:, 1]]
    loss2 = product[triplets[:, 2], triplets[:, 3]]
    return loss1.mean() + loss2.mean(), product.max()


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

    print(args)

    # Initialize prototypes and optimizer.
    if os.path.exists(args.wtvfile):
        print(f"initialiing prototypes with w2vec file {args.wtvfile}")
        use_wtv = True
        wtvv = torch.from_numpy(np.load(args.wtvfile)).float().to(device)
        wtvv = F.normalize(wtvv, p=2, dim=1)
        wtvsim = torch.matmul(wtvv, wtvv.t()).float()

        # Precompute triplets.
        nns, others = [], []
        for i in range(wtvv.shape[0]):
            sorder = torch.argsort(wtvsim[i, :], descending=True)
            nns.append(sorder[:args.nn])
            others.append(sorder[args.nn:-1])
        triplets = []
        for i in range(wtvv.shape[0]):
            for j in nns[i]:
                for k in others[i]:
                    triplets.append([i, j, i, k])
        triplets = torch.tensor(triplets, device=device, dtype=torch.int)
    else:
        use_wtv = False

    # Initialize prototypes.
    prototypes = torch.randn(args.classes, args.dims, device=device)
    prototypes = geoopt.ManifoldParameter(F.normalize(prototypes, p=2, dim=1), manifold=geoopt.SphereExact())
    if args.optim == "rsgd":
        optimizer = geoopt.optim.RiemannianSGD([prototypes], lr=args.learning_rate, momentum=args.momentum, stabilize=1)
    elif args.optim == "radam":
        optimizer = geoopt.optim.RiemannianAdam([prototypes], lr=args.learning_rate, stabilize=1)
    else:
        raise ValueError("Unknown optimizer")
    # Optimize for separation.
    for i in range(args.epochs):
        # Compute loss.
        loss1, sep = prototype_loss(prototypes)
        if use_wtv:
            loss2, _ = prototype_loss_sem(prototypes, triplets)
            loss = loss1 + loss2
        else:
            loss = loss1

        if args.dispersion=='mmd':
            disp_loss = ledoh_torch.kernel_dispersion.KernelSphereDispersion().forward(prototypes)
        elif args.dispersion=='lloyd':
            disp_loss = ledoh_torch.lloyd_dispersion.LloydSphereDispersion().forward(prototypes)[0]
        elif args.dispersion=='sliced':
            disp_loss = ledoh_torch.sliced_dispersion.SlicedSphereDispersion().forward(prototypes, p=None, q=None)
        else:
            disp_loss = None
        # Update.

        if disp_loss is not None:
            loss += args.dispersion_weight * disp_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        c_time = time.strftime("%d %m %Y %H:%M:%S",time.localtime())
        print(c_time," ", "%03d/%d: %.4f\r" % (i, args.epochs, sep))

    # Store result.
    os.makedirs(args.resdir, exist_ok=True)
    if args.dispersion is not None:
        fn = f"{args.resdir}/prototypes-{args.dims}d-{args.classes}c-{args.dispersion}-{args.dispersion_weight}dw.npy"
    else:
        fn = f"{args.resdir}/prototypes-{args.dims}d-{args.classes}c.npy"
    np.save(fn, prototypes.cpu().data.numpy())
