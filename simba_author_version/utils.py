import torch
import numpy as np
import torchvision.transforms as trans
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms, models

import math
from scipy.fftpack import dct, idct

# mean and std for different datasets
IMAGENET_SIZE = 224
# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_MEAN = [0, 0, 0]
IMAGENET_STD = [1, 1, 1]
# IMAGENET_TRANSFORM = trans.Compose([
#     trans.Scale(256),
#     trans.CenterCrop(224),
#     trans.ToTensor()])

# INCEPTION_SIZE = 299
# INCEPTION_TRANSFORM = trans.Compose([
#     trans.Scale(342),
#     trans.CenterCrop(299),
#     trans.ToTensor()])

CIFAR_SIZE = 32
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]
# CIFAR_TRANSFORM = trans.Compose([
#     trans.ToTensor()])

# MNIST_SIZE = 28
# MNIST_MEAN = [0.5]
# MNIST_STD = [1.0]
# MNIST_TRANSFORM = trans.Compose([
#     trans.ToTensor()])



# reverses the normalization transformation
def invert_normalization(imgs, dataset):
    if dataset == 'tinyimagenet':
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    elif dataset == 'cifar':
        mean = CIFAR_MEAN
        std = CIFAR_STD
    elif dataset == 'mnist':
        mean = MNIST_MEAN
        std = MNIST_STD
    imgs_trans = imgs.clone()
    if len(imgs.size()) == 3:
        for i in range(imgs.size(0)):
            imgs_trans[i, :, :] = imgs_trans[i, :, :] * std[i] + mean[i]
    else:
        for i in range(imgs.size(1)):
            imgs_trans[:, i, :, :] = imgs_trans[:, i, :, :] * std[i] + mean[i]
    return imgs_trans


# applies the normalization transformations
def apply_normalization(imgs, dataset):
    if dataset == 'tinyimagenet':
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    elif dataset == 'cifar':
        mean = CIFAR_MEAN
        std = CIFAR_STD
    elif dataset == 'mnist':
        mean = MNIST_MEAN
        std = MNIST_STD
    else:
        mean = [0, 0, 0]
        std = [1, 1, 1]
    imgs_tensor = imgs.clone()
    if dataset == 'mnist':
        imgs_tensor = (imgs_tensor - mean[0]) / std[0]
    else:
        if imgs.dim() == 3:
            for i in range(imgs_tensor.size(0)):
                imgs_tensor[i, :, :] = (imgs_tensor[i, :, :] - mean[i]) / std[i]
        else:
            for i in range(imgs_tensor.size(1)):
                imgs_tensor[:, i, :, :] = (imgs_tensor[:, i, :, :] - mean[i]) / std[i]
    return imgs_tensor


# get most likely predictions and probabilities for a set of inputs
def get_preds(model, inputs, dataset_name, correct_class=None, batch_size=25, return_cpu=True):
    num_batches = int(math.ceil(inputs.size(0) / float(batch_size)))
    softmax = torch.nn.Softmax()
    all_preds, all_probs = None, None
    transform = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    for i in range(num_batches):
        upper = min((i + 1) * batch_size, inputs.size(0))
        input = apply_normalization(inputs[(i * batch_size):upper], dataset_name)
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)
        output = softmax.forward(model.forward(input_var))
        if correct_class is None:
            prob, pred = output.max(1)
        else:
            prob, pred = output[:, correct_class], torch.autograd.Variable(torch.ones(output.size()) * correct_class)
        if return_cpu:
            prob = prob.data.cpu()
            pred = pred.data.cpu()
        else:
            prob = prob.data
            pred = pred.data
        if i == 0:
            all_probs = prob
            all_preds = pred
        else:
            all_probs = torch.cat((all_probs, prob), 0)
            all_preds = torch.cat((all_preds, pred), 0)
    return all_preds, all_probs


# get least likely predictions and probabilities for a set of inputs
def get_least_likely(model, inputs, dataset_name, batch_size=25, return_cpu=True):
    num_batches = int(math.ceil(inputs.size(0) / float(batch_size)))
    softmax = torch.nn.Softmax()
    all_preds, all_probs = None, None
    transform = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    for i in range(num_batches):
        upper = min((i + 1) * batch_size, inputs.size(0))
        input = apply_normalization(inputs[(i * batch_size):upper], dataset_name)
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)
        output = softmax.forward(model.forward(input_var))
        prob, pred = output.min(1)
        if return_cpu:
            prob = prob.data.cpu()
            pred = pred.data.cpu()
        else:
            prob = prob.data
            pred = pred.data
        if i == 0:
            all_probs = prob
            all_preds = pred
        else:
            all_probs = torch.cat((all_probs, prob), 0)
            all_preds = torch.cat((all_preds, pred), 0)
    return all_preds, all_probs


# defines a diagonal order
# order is fixed across diagonals but are randomized across channels and within the diagonal
# e.g.
# [1, 2, 5]
# [3, 4, 8]
# [6, 7, 9]
def diagonal_order(image_size, channels):
    x = torch.arange(0, image_size).cumsum(0)
    order = torch.zeros(image_size, image_size)
    for i in range(image_size):
        order[i, :(image_size - i)] = i + x[i:]
    for i in range(1, image_size):
        reverse = order[image_size - i - 1].index_select(0, torch.LongTensor([i for i in range(i-1, -1, -1)]))
        order[i, (image_size - i):] = image_size * image_size - 1 - reverse
    if channels > 1:
        order_2d = order
        order = torch.zeros(channels, image_size, image_size)
        for i in range(channels):
            order[i, :, :] = 3 * order_2d + i
    return order.view(1, -1).squeeze().long().sort()[1]


# defines a block order, starting with top-left (initial_size x initial_size) submatrix
# expanding by stride rows and columns whenever exhausted
# randomized within the block and across channels
# e.g. (initial_size=2, stride=1)
# [1, 3, 6]
# [2, 4, 9]
# [5, 7, 8]
def block_order(image_size, channels, initial_size=1, stride=1):
    order = torch.zeros(channels, image_size, image_size)
    total_elems = channels * initial_size * initial_size
    perm = torch.randperm(total_elems)
    order[:, :initial_size, :initial_size] = perm.view(channels, initial_size, initial_size)
    for i in range(initial_size, image_size, stride):
        num_elems = channels * (2 * stride * i + stride * stride)
        perm = torch.randperm(num_elems) + total_elems
        num_first = channels * stride * (stride + i)
        order[:, :(i+stride), i:(i+stride)] = perm[:num_first].view(channels, -1, stride)
        order[:, i:(i+stride), :i] = perm[num_first:].view(channels, stride, -1)
        total_elems += num_elems
    return order.view(1, -1).squeeze().long().sort()[1]


# zeros all elements outside of the top-left (block_size * ratio) submatrix for every block
def block_zero(x, block_size=8, ratio=0.5):
    z = torch.zeros(x.size())
    num_blocks = int(x.size(2) / block_size)
    mask = torch.zeros(x.size(0), x.size(1), block_size, block_size)
    mask[:, :, :int(block_size * ratio), :int(block_size * ratio)] = 1
    for i in range(num_blocks):
        for j in range(num_blocks):
            z[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)] = x[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)] * mask
    return z


# applies DCT to each block of size block_size
def block_dct(x, block_size=8, masked=False, ratio=0.5):
    z = torch.zeros(x.size())
    num_blocks = int(x.size(2) / block_size)
    mask = np.zeros((x.size(0), x.size(1), block_size, block_size))
    mask[:, :, :int(block_size * ratio), :int(block_size * ratio)] = 1
    for i in range(num_blocks):
        for j in range(num_blocks):
            submat = x[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)].numpy()
            submat_dct = dct(dct(submat, axis=2, norm='ortho'), axis=3, norm='ortho')
            if masked:
                submat_dct = submat_dct * mask
            submat_dct = torch.from_numpy(submat_dct)
            z[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)] = submat_dct
    return z


# applies IDCT to each block of size block_size
def block_idct(x, block_size=8, masked=False, ratio=0.5, linf_bound=0.0):
    z = torch.zeros(x.size())
    num_blocks = int(x.size(2) / block_size)
    mask = np.zeros((x.size(0), x.size(1), block_size, block_size))
    if type(ratio) != float:
        for i in range(x.size(0)):
            mask[i, :, :int(block_size * ratio[i]), :int(block_size * ratio[i])] = 1
    else:
        mask[:, :, :int(block_size * ratio), :int(block_size * ratio)] = 1
    for i in range(num_blocks):
        for j in range(num_blocks):
            submat = x[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)].numpy()
            if masked:
                submat = submat * mask
            z[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)] = torch.from_numpy(idct(idct(submat, axis=3, norm='ortho'), axis=2, norm='ortho'))
    if linf_bound > 0:
        return z.clamp(-linf_bound, linf_bound)
    else:
        return z

def load_checkpoint(ckpt_path, map_location='cpu'):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt


def get_dataset(dataset ,transform = None, train_split=0.8):

    if dataset == 'cifar':
        if transform == None:
            print('Specify a transformation function')
        
        train_data = datasets.CIFAR10('./cifar', train=True, transform = transform['train_transform'], download=True)
        # num_training_imgs = int(len(training_data) * train_split)
        # torch.manual_seed(0)
        # train_data, val_data = torch.utils.data.random_split(training_data, [num_training_imgs, len(training_data) - num_training_imgs])

        test_data = datasets.CIFAR10('./cifar', train=False, transform = transform['test_transform'], download=True)

        return {'train_data': train_data,
                'test_data': test_data}

    if dataset == 'tinyimagenet':
        import tiny_imagenet
        trainset = tiny_imagenet.TinyImageNet200(root='./data', train= True, transform=transform['train_transform'],download=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=True, num_workers=2)

        testset = tiny_imagenet.TinyImageNet200(root='./data', train=False, transform=transform['test_transform'])
        testloader = torch.utils.data.DataLoader(testset, batch_size=25, shuffle=False, num_workers=2)

        print( "trainset num = {}".format(len(trainset)) )
        print( "trainloader num = {}".format(len(testset)) )
        return {'train_data': trainset,
                'test_data': testset}


def get_transformation(dataset):
    
    if dataset == 'cifar':

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])


        transform = {'train_transform': train_transform,
                     'test_transform': test_transform}


    if dataset == 'tinyimagenet':
        test_transform = transforms.Compose([transforms.ToTensor()])
        transform = {'train_transform': test_transform,
                     'val_transform': test_transform,
                     'test_transform': test_transform}

    else:
        print('Please choose the right dataset!!!')

    return transform


# def load_model(model_path, basic_net):
#     checkpoint = torch.load(model_path)
#     basic_net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']
    
#     return best_acc, start_epoch


def load_model(model_path, basic_net):
    checkpoint = torch.load(model_path)
    
    #print(checkpoint)
    
    #basic_net.load_state_dict(checkpoint['net'])
    basic_net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['net'].items()})
    # best_acc_nat = checkpoint['acc_clean']
    # best_acc_l2 = checkpoint['acc_l2']
    # best_acc_linf = checkpoint['acc_linf']
    
    # return best_acc_nat, best_acc_l2, best_acc_linf


def load_model_nat(model_path, basic_net):
    checkpoint = torch.load(model_path)
    
    #print(checkpoint)
    
    #basic_net.load_state_dict(checkpoint['net'])
    basic_net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['net'].items()})
    best_acc_nat = checkpoint['acc_clean']
    best_acc_fog = checkpoint['acc_fog']
    best_acc_snow = checkpoint['acc_snow']    
    
    return best_acc_nat, best_acc_fog, best_acc_snow
