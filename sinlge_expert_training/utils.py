import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, transforms, models

def cuda(xs, gpu_id):
    if torch.cuda.is_available():
        if not isinstance(xs, (list, tuple)):
            return xs.cuda(int(gpu_id[0]))
        else:
            return [x.cuda(int(gpu_id[0])) for x in xs]
    return xs

def load_checkpoint(ckpt_path, map_location='cpu'):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt


def get_dataset(dataset ,transform = None, train_split=0.8):

    if dataset == 'cifar':
        if transform == None:
            print('Specify a transformation function')
        
        training_data = datasets.CIFAR10('./cifar', train=True, transform = transform['train_transform'], download=True)
        num_training_imgs = int(len(training_data) * train_split)
        torch.manual_seed(0)
        train_data, val_data = torch.utils.data.random_split(training_data, [num_training_imgs, len(training_data) - num_training_imgs])

        test_data = datasets.CIFAR10('./cifar', train=False, transform = transform['test_transform'], download=True)

        return {'train_data': train_data,
                'val_data': val_data,
                'test_data': test_data}
    if dataset == 'tinyimagenet':
        import tiny_imagenet
        trainset = tiny_imagenet.TinyImageNet200(root='./data', train= True, transform=transform['train_transform'],download=True)
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=True, num_workers=2)

        testset = tiny_imagenet.TinyImageNet200(root='./data', train=False, transform=transform['test_transform'])
        # testloader = torch.utils.data.DataLoader(testset, batch_size=25, shuffle=False, num_workers=2)

        print( "trainset num = {}".format(len(trainset)) )
        print( "trainloader num = {}".format(len(testset)) )
        return {'train_data': trainset,
                'val_data': testset,
                'test_data': testset}
    # if dataset == 'tinyimagenet':
    #     import tinyprocess
    #     return {'train_data': tinyprocess.train_dataset,
    #             'val_data': tinyprocess.train_dataset,
    #             }


        # train_dir = '../tiny-imagenet-200/train'
        # val_dir = '../tiny-imagenet-200/val'
        # test_dir = '../tiny-imagenet-200/test'
        # train_data = datasets.ImageFolder(train_dir, transform=transforms.ToTensor())
        # val_data = datasets.ImageFolder(val_dir, transform=transforms.ToTensor())
        # test_data = datasets.ImageFolder(test_dir, transform=transforms.ToTensor())
        # # train_loader = torch.utils.data.DataLoader(train_data, batch_size=128)
        # # val_loader = torch.utils.data.DataLoader(val_data, batch_size=128)
        # # test_loader = torch.utils.data.DataLoader(test_data, batch_size=128)
        # return {'train_data': train_data,
        # 'val_data': val_data,
        # 'test_data': test_data,
        # # 'train_loader': train_loader,
        # # 'val_loader': val_loader,
        # # 'test_loader':test_loader
        # }


def get_transformation(dataset):
    
    if dataset == 'cifar':
        # train_transform = transforms.Compose([transforms.RandomResizedCrop(size=28),
        #                                       transforms.ToTensor(),
        #                                       transforms.Normalize([0.485, 0.456, 0.406],
        #                                                            [0.229, 0.224, 0.225])])

        # test_transform = transforms.Compose([transforms.Resize(size=28),
        #                                       transforms.ToTensor(),
        #                                       transforms.Normalize([0.485, 0.456, 0.406],
        #                                                            [0.229, 0.224, 0.225])])
        train_transform = transforms.Compose([
                                                transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ColorJitter(.25,.25,.25),
                                                transforms.RandomRotation(2),
                                                transforms.ToTensor(),
                                              ])

        test_transform = transforms.Compose([
                                                transforms.ToTensor(),
                                              ])

        transform = {'train_transform': train_transform,
                    'val_transform': train_transform,
                     'test_transform': test_transform}
    elif dataset=='tinyimagenet':

        train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=64),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(.25,.25,.25),
        transforms.RandomRotation(2),
        transforms.ToTensor(),
                                              ])

        test_transform = transforms.Compose([
        transforms.ToTensor(),
                                              ])        
        transform = {'train_transform': train_transform,
                    'val_transform': train_transform,
                     'test_transform': test_transform}

    else:
        raise Exception('Please choose the right dataset!!!')

    return transform


def load_model(model_path, basic_net):
    checkpoint = torch.load(model_path)
    
    #print(checkpoint)
    
    #basic_net.load_state_dict(checkpoint['net'])
    basic_net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['net'].items()})
    best_acc_nat = checkpoint['acc_clean']
    best_acc_l2 = checkpoint['acc_l2']
    best_acc_linf = checkpoint['acc_linf']    
    return best_acc_nat, best_acc_l2, best_acc_linf

def EM_loss():
    '''
    I am still not able to get the notation specified in the paper for calculating EM loss as it is a bit confusing how they have
    calculated the posterior probabilities in equation (5).
    Also the updation step for the expert and gating weights they have used is just simply iteration based rather than gradient based which
    is also pretty weird.

    Also Cross Entropy would also work in this particular case as the model is a classification model and
    I think that the gradient based method also solves the MLE optimization problem(for which EM algorithm is suggested) as specified in the paper.
    '''