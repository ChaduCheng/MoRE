import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import numpy as np
import utils
import math
import random
import torch.nn.functional as F
import argparse
import os
import pdb
from model_resnet import ResNet18, MoE_ResNet18

parser = argparse.ArgumentParser(description='Runs SimBA on a set of images')
# parser.add_argument('--data_root', type=str, required=True, help='root directory of imagenet data')
# parser.add_argument('--result_dir', type=str, default='save', help='directory for saving results')
# parser.add_argument('--sampled_image_dir', type=str, default='save', help='directory to cache sampled images')
# parser.add_argument('--model', type=str, default='resnet50', help='type of base model to use')
# parser.add_argument('--num_runs', type=int, default=1000, help='number of image samples')
# parser.add_argument('--batch_size', type=int, default=50, help='batch size for parallel runs')
# parser.add_argument('--num_iters', type=int, default=0, help='maximum number of iterations, 0 for unlimited')
# parser.add_argument('--log_every', type=int, default=10, help='log every n iterations')
# parser.add_argument('--epsilon', type=float, default=0.2, help='step size per iteration')
# parser.add_argument('--linf_bound', type=float, default=0.0, help='L_inf bound for frequency space attack')
# parser.add_argument('--freq_dims', type=int, default=14, help='dimensionality of 2D frequency space')
# parser.add_argument('--order', type=str, default='rand', help='(random) order of coordinate selection')
# parser.add_argument('--stride', type=int, default=7, help='stride for block order')
# parser.add_argument('--targeted', action='store_true', help='perform targeted attack')
# parser.add_argument('--pixel_attack', action='store_true', help='attack in pixel space')
# parser.add_argument('--save_suffix', type=str, default='', help='suffix appended to save file')
# args = parser.parse_args()
# args.dataset = 'cifar'
# #arags.data_root = ''
# args.result_dir = './results'
# args.sampled_image_dir = ''
# args.model = 'resnet18'
# args.num_runs = 1000
# args.batch_size = 8
# args.num_iters = 0
# args.log_every = 10
# args.epsilon = 0.314*5
# args.linf_bound = 0.0
# args.freq_dims = 14
# args.order = ''  # only used in frequency attack
# args.stride = 7
# args.targeted = False
# args.pixel_attack = True
# args.save_suffix = ''
# args.train_split = 0.8


def expand_vector(x, size, image_size):  # 在frequnecy 里有着明确的意义
    batch_size = x.size(0)
    x = x.view(-1, 3, size, size)
    z = torch.zeros(batch_size, 3, image_size, image_size)
    z[:, :, :size, :size] = x
    return z

# def normalize(x):
#     return utils.apply_normalization(x, 'imagenet')

def get_probs(model, x, y):
    #output = model(normalize(torch.autograd.Variable(x.cuda()))).cpu()
    x = x.cuda()
    output = model(torch.autograd.Variable(x)).cpu()
    #output = output.cuda()
    a = torch.nn.Softmax()(output).data
    probs = torch.index_select(a, 1, y)   # change here
    return torch.diag(probs)

def get_preds(model, x):   # obtain the prediction 
    output = model(torch.autograd.Variable(x.cuda())).cpu()
    _, preds = output.data.max(1)
    return preds

def get_preds_true(model, x):   # obtain the prediction
    output = model(x.cuda()).cpu()
    _, preds = output.data.max(1)
    return preds

# runs simba on a batch of images <images_batch> with true labels (for untargeted attack) or target labels
# (for targeted attack) <labels_batch>
def dct_attack_batch(model, images_batch, labels_batch, max_iters, freq_dims, stride, epsilon, order='rand', targeted=False, pixel_attack=False, log_every=1):
    batch_size = images_batch.size(0)   # 100 / 1000
    image_size = images_batch.size(2)   # 224
    # sample a random ordering for coordinates independently per batch element
    if order == 'rand':   # random obain
        indices = torch.randperm(3 * freq_dims * freq_dims)[:max_iters]
    elif order == 'diag':
        indices = utils.diagonal_order(image_size, 3)[:max_iters]
    elif order == 'strided':
        indices = utils.block_order(image_size, 3, initial_size=freq_dims, stride=stride)[:max_iters]
    else:
        indices = utils.block_order(image_size, 3)[:max_iters]
    if order == 'rand':   # give the input size
        expand_dims = freq_dims
    else:
        expand_dims = image_size
    n_dims = 3 * expand_dims * expand_dims
    x = torch.zeros(batch_size, n_dims)
    # logging tensors
    probs = torch.zeros(batch_size, max_iters)
    succs = torch.zeros(batch_size, max_iters)
    queries = torch.zeros(batch_size, max_iters)
    l2_norms = torch.zeros(batch_size, max_iters)
    linf_norms = torch.zeros(batch_size, max_iters)
    prev_probs = get_probs(model, images_batch, labels_batch)
    preds = get_preds(model, images_batch)
    if pixel_attack:
        trans = lambda z: z
    else:
        trans = lambda z: utils.block_idct(z, block_size=image_size, linf_bound=args.linf_bound)
    remaining_indices = torch.arange(0, batch_size).long()  # 0-batch_size 的大小
    for k in range(max_iters):
        dim = indices[k]
        expanded = (images_batch[remaining_indices] + trans(expand_vector(x[remaining_indices], expand_dims, image_size))).clamp(0, 1)
        perturbation = trans(expand_vector(x, expand_dims,image_size))
        l2_per = perturbation.view(batch_size, -1).norm(2, 1)
        l2_norms[:, k] = l2_per    # compute l2 norm
        linf_per = perturbation.view(batch_size, -1).abs().max(1)[0]
        linf_norms[:, k] =  linf_per  # comput linf norm
        preds_next = get_preds(model, expanded)
        preds[remaining_indices] = preds_next

        # decide how many item are same with each other
        if targeted:
            remaining = preds.ne(labels_batch)
        else:
            remaining = preds.eq(labels_batch)
        # check if all images are misclassified and stop early
        if remaining.sum() == 0:
            adv = (images_batch + trans(expand_vector(x, expand_dims,image_size))).clamp(0, 1)
            probs_k = get_probs(model, adv, labels_batch)
            probs[:, k:] = probs_k.unsqueeze(1).repeat(1, max_iters - k)
            succs[:, k:] = torch.ones(args.batch_size, max_iters - k)
            queries[:, k:] = torch.zeros(args.batch_size, max_iters - k)
            break
        remaining_indices = torch.arange(0, batch_size)[remaining].long()
        if k > 0:
            succs[:, k-1] = ~remaining


        diff = torch.zeros(remaining.sum(), n_dims)
        diff[:, dim] = epsilon
        left_vec = x[remaining_indices] - diff
        right_vec = x[remaining_indices] + diff
        # trying negative direction
        adv = (images_batch[remaining_indices] + trans(expand_vector(left_vec, expand_dims, image_size))).clamp(0, 1)
        left_probs = get_probs(model, adv, labels_batch[remaining_indices])
        queries_k = torch.zeros(batch_size)
        # increase query count for all images
        queries_k[remaining_indices] += 1
        if targeted:
            improved = left_probs.gt(prev_probs[remaining_indices])
        else:
            improved = left_probs.lt(prev_probs[remaining_indices])
        # only increase query count further by 1 for images that did not improve in adversarial loss
        if improved.sum() < remaining_indices.size(0):
            queries_k[remaining_indices[~improved]] += 1
        # try positive directions
        adv = (images_batch[remaining_indices] + trans(expand_vector(right_vec, expand_dims, image_size))).clamp(0, 1)
        right_probs = get_probs(model, adv, labels_batch[remaining_indices])
        if targeted:
            right_improved = right_probs.gt(torch.max(prev_probs[remaining_indices], left_probs))
        else:
            right_improved = right_probs.lt(torch.min(prev_probs[remaining_indices], left_probs))
        probs_k = prev_probs.clone()
        # update x depending on which direction improved
        if improved.sum() > 0:  # left improved
            left_indices = remaining_indices[improved]
            left_mask_remaining = improved.unsqueeze(1).repeat(1, n_dims)
            x[left_indices] = left_vec[left_mask_remaining].view(-1, n_dims)
            probs_k[left_indices] = left_probs[improved]
        if right_improved.sum() > 0:
            right_indices = remaining_indices[right_improved]
            right_mask_remaining = right_improved.unsqueeze(1).repeat(1, n_dims)
            x[right_indices] = right_vec[right_mask_remaining].view(-1, n_dims)
            probs_k[right_indices] = right_probs[right_improved]
        probs[:, k] = probs_k
        queries[:, k] = queries_k
        prev_probs = probs[:, k]
        if (k + 1) % log_every == 0 or k == max_iters - 1:
            print('Iteration %d: queries = %.4f, prob = %.4f, succ = %.4f, remaining = %.4f' % (
                    k + 1, queries.sum(1).mean(), probs[:, k].mean(), succs[:, k].mean(), remaining.float().mean()))
    expanded = (images_batch + trans(expand_vector(x, expand_dims, image_size))).clamp(0, 1)
    preds = get_preds(model, expanded)
    if targeted:
        remaining = preds.ne(labels_batch)
    else:
        remaining = preds.eq(labels_batch)
    succs[:, max_iters-1] = ~remaining
    return expanded, probs, succs, queries, l2_norms, linf_norms

