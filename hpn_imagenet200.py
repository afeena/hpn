#
# Perform classification on ImageNet-200 using Hypershperical
# Prototype Networks.
#
# @inproceedings{mettes2016hyperspherical,
#  title={Hyperspherical Prototype Networks},
#  author={Mettes, Pascal and van der Pol, Elise and Snoek, Cees G M},
#  booktitle={Advances in Neural Information Processing Systems},
#  year={2019}
# }
#

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

import helper

sys.path.append("models/imagenet200/")
from models.imagenet200 import densenet, resnet


################################################################################
# Training epoch.
################################################################################

#
# Main training function.
#
# model (object)      - Network.
# device (torch)      - Torch device, e.g. CUDA or CPU.
# trainloader (torch) - Training data.
# optimizer (torch)   - Type of optimizer.
# f_loss (torch)      - Loss function.
# epoch (int)         - Epoch iteration.
#
def main_train(model, device, trainloader, optimizer, f_loss, epoch):
    # Set mode to training.
    model.train()
    avgloss, avglosscount = 0., 0.

    # Go over all batches.
    for bidx, (data, target) in enumerate(trainloader):
        # Data to device.
        target = model.polars[target]
        data = torch.autograd.Variable(data).cuda()
        target = torch.autograd.Variable(target).cuda()

        # Compute outputs and losses.
        output = model(data)
        loss = (1 - f_loss(output, target)).pow(2).sum()

        # Backpropagation.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update loss.
        avgloss += loss.item()
        avglosscount += 1.
        newloss = avgloss / avglosscount

        # Print updates.
        c_time = time.strftime("%d %m %Y %H:%M:%S", time.localtime())
        print(c_time, " Training epoch %d: loss %8.4f - %.0f\r" \
              % (epoch, newloss, 100. * (bidx + 1) / len(trainloader)))
    print()


################################################################################
# Testing epoch.
################################################################################

#
# Main test function.
#
# model (object)             - Network.
# device (torch)             - Torch device, e.g. CUDA or CPU.
# testloader (torch)         - Test data.
#
def main_test(model, device, testloader, epoch, hpnfile, save_folder=None):
    # Set model to evaluation and initialize accuracy and cosine similarity.
    model.eval()
    cos = nn.CosineSimilarity(eps=1e-9)
    acc = 0

    # Go over all batches.
    with torch.no_grad():
        results = []
        for data, target in testloader:
            # Data to device.
            data = torch.autograd.Variable(data).cuda()
            target = target.cuda()
            target = torch.autograd.Variable(target)

            # Forward.
            output = model(data).float()
            output = model.predict(output).float()

            pred = output.max(1, keepdim=True)[1]
            results.append(np.concatenate((pred.cpu().numpy(), target.view_as(pred).cpu().numpy()), axis=1))
            acc += pred.eq(target.view_as(pred)).sum().item()

    results = np.concatenate(results, axis=0)
    # Print results.
    testlen = len(testloader.dataset)
    hpfile = hpnfile.split("/")[-1]
    if save_folder is not None:
        with open(f"{save_folder}/test_acc.log", "a") as f:
            f.write(f"-#@x- Epoch: {epoch} | Accuracy: {100. * acc / testlen}, | Hpnfile: {hpfile} -#@x-\n")
        np.save(f"{save_folder}/test_epoch_{epoch}", results)
    else:
        print(f"Epoch: {epoch} | Accuracy: {100. * acc / testlen}, | Hpnfile: {hpfile}")
    return acc / float(testlen)


################################################################################
# Main entry point of the script.
################################################################################

#
# Parse all user arguments.
#
def parse_args():
    parser = argparse.ArgumentParser(description="IMAGENET-200 classification")
    parser.add_argument("--datadir", dest="datadir", default="dat/", type=str)
    parser.add_argument("--resdir", dest="resdir", default="res/", type=str)
    parser.add_argument("--hpnfile", dest="hpnfile", default="", type=str)
    parser.add_argument("-n", dest="network", default="resnet32", type=str)
    parser.add_argument("-r", dest="optimizer", default="sgd", type=str)
    parser.add_argument("-l", dest="learning_rate", default=0.01, type=float)
    parser.add_argument("-m", dest="momentum", default=0.9, type=float)
    parser.add_argument("-c", dest="decay", default=0.0001, type=float)
    parser.add_argument("-s", dest="batch_size", default=128, type=int)
    parser.add_argument("-e", dest="epochs", default=250, type=int)
    parser.add_argument("--drop1", dest="drop1", default=100, type=int)
    parser.add_argument("--drop2", dest="drop2", default=200, type=int)
    parser.add_argument("--seed", dest="seed", default=100, type=int)
    args = parser.parse_args()
    return args


#
# Main entry point of the script.
#
if __name__ == "__main__":
    # Parse user parameters and set device.
    args = parse_args()
    device = torch.device("cuda")
    kwargs = {'num_workers': 32, 'pin_memory': True}
    args.dataset = "imagenet200"

    # Set the random seeds.
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if args.resdir is not None:
        import hashlib
        from datetime import datetime
        h = hashlib.shake_256()
        save_str = str(args.hpnfile) + " "+str(datetime.now())
        h.update(save_str.encode())
        save_folder = f"{args.resdir}/{h.hexdigest(10)}"
    else:
        save_folder = None

    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        with open(f"{save_folder}/config.json", "w") as f:
                json.dump(args.__dict__, f, indent=2)
    # Load data.
    batch_size = args.batch_size
    trainloader, testloader = helper.load_imagenet200(args.datadir, \
                                                      batch_size, kwargs)
    nr_classes = 200

    # Load the polars and update the trainy labels.
    classpolars = torch.from_numpy(np.load(args.hpnfile)).float()
    args.output_dims = int(args.hpnfile.split("/")[-1].split("-")[1][:-1])

    # Load the model.
    if args.network == "resnet32":
        model = resnet.ResNet(32, args.output_dims, 1, classpolars)
    elif args.network == "densenet121":
        model = densenet.DenseNet121(args.output_dims, classpolars)
    model = model.to(device)

    # Load the optimizer.
    optimizer = helper.get_optimizer(args.optimizer, model.parameters(), \
                                     args.learning_rate, args.momentum, args.decay)

    # Initialize the loss functions.
    f_loss = nn.CosineSimilarity(eps=1e-9).cuda()

    # Main loop.
    testscores = []
    learning_rate = args.learning_rate
    for i in range(args.epochs):
        print("---")
        # Learning rate decay.
        if i in [args.drop1, args.drop2]:
            learning_rate *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        # Train and test.
        main_train(model, device, trainloader, optimizer, f_loss, i)
        if i % 100 == 0 or i == args.epochs - 1:
            t = main_test(model, device, testloader, i, args.hpnfile, save_folder=save_folder)
            testscores.append([i, t])
            if save_folder is not None:
                torch.save(model.state_dict(), f"{args.resdir}/model_epoch_{i}.pt")
