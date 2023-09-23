"""Contains losses defined as per DeepVesselNet paper by Giles Tetteh"""

import torch
from torch.nn import functional as F
import numpy as np

from dvn import misc as ms

def dice_loss(output, target):
    """
    output is a torch variable of size BatchxCxHxWxD representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    print('[dice_loss]', '[output]', output.dtype, output.shape, output.min(), output.max(), output.sum())
    print('[dice_loss]', '[target]', target.dtype, target.shape, target.min(), target.max())
    target = F.one_hot(target.long(), num_classes=2).permute(0, 4, 1, 2, 3)
    print('[dice_loss]', '[target]', target.dtype, target.shape, target.min(), target.max())

    probs = F.softmax(output, dim=1)
    # probs = output
    print('[dice_loss]', '[probs]', probs.dtype, probs.shape, probs.min(), probs.max())
    num = probs * target  # b,c,h,w--p*g
    print('[dice_loss]', '[num]', num.dtype, num.shape, num.min(), num.max())
    num = torch.sum(num, dim=4)  # b,c,h
    print('[dice_loss]', '[num]', num.dtype, num.shape, num.min(), num.max())
    num = torch.sum(num, dim=3)
    print('[dice_loss]', '[num]', num.dtype, num.shape, num.min(), num.max())
    num = torch.sum(num, dim=2)
    print('[dice_loss]', '[num]', num)

    den1 = probs * probs  # --p^2
    print('[dice_loss]', '[den1]', den1.dtype, den1.shape, den1.min(), den1.max())
    den1 = torch.sum(probs, dim=4)  # b,c,h
    print('[dice_loss]', '[den1]', den1.dtype, den1.shape, den1.min(), den1.max())
    den1 = torch.sum(den1, dim=3)
    print('[dice_loss]', '[den1]', den1.dtype, den1.shape, den1.min(), den1.max())
    den1 = torch.sum(den1, dim=2)
    print('[dice_loss]', '[den1]', den1)

    den2 = target * target  # --g^2
    print('[dice_loss]', '[den2]', den2.dtype, den2.shape, den2.min(), den2.max())
    den2 = torch.sum(target, dim=4)  # b,c,h
    print('[dice_loss]', '[den2]', den2.dtype, den2.shape, den2.min(), den2.max())
    den2 = torch.sum(den2, dim=3)  # b,c
    print('[dice_loss]', '[den2]', den2.dtype, den2.shape, den2.min(), den2.max())
    den2 = torch.sum(den2, dim=2)  # b,c
    print('[dice_loss]', '[den2]', den2)

    dice = 2 * (num / (den1 + den2))
    print('[dice_loss]', '[dice]', dice)
    dice_eso = 1 - dice[:, 1:]  # we ignore bg dice val, and take the fg
    print('[dice_loss]', '[dice_eso]', dice_eso)

    dice_total = torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz
    print('[dice_loss]', '[dice_total]', dice_total)

    return dice_total


if __name__ == '__main__':
    # volume, target = train_synthetic[0]
    # volume, target = volume.unsqueeze(0).to(device, dtype=torch.float), target.unsqueeze(0).to(device, dtype=torch.long)
    volume, target = next(iter(train_loader))
    output = model(volume.to(device))
    print(volume.shape, target.shape, output.shape)
    dice_loss(output, target.to(device))
