#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 15:54:37 2020

@author: chad
"""

#import os
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import optim
import torch.backends.cudnn as cudnn

#import utils
#from model_adv import AlexNet, MoE_alexnet
from model_adv_att import AttackPGD
#from model_resnet import *
import string

from torch.utils.data import DataLoader

import numpy as np
import utils
import math
import random
# import torch.nn.functional as F
# import argparse
import os
import pdb
from model_resnet import ResNet18, MoE_ResNet18
from run_simba_cifar import dct_attack_batch









def test(args):

    config_linf = {
    'epsilon': 8.0 / 255,
    #'epsilon': 0.314,
    'num_steps': 10,
    'step_size': 2.0 / 255,
    'random_start': True,
    'loss_func': 'xent',
    '_type': 'linf'
     }
    
    config_l2 = {
    #'epsilon': 8.0 / 255,
    'epsilon': 0.314,
    'num_steps': 10,
    'step_size': 2.0 / 255,
    'random_start': True,
    'loss_func': 'xent',
    '_type': 'l2'
     }
    
    attack = 'true'

    if args.dataset == 'cifar':
        output_classes = 10
        
    global best_acc_nat, best_acc_l2, best_acc_linf
    best_acc_nat = 0
    best_acc_l2 = 0
    best_acc_linf = 0
    

    transform = utils.get_transformation(args.dataset)
    dataset = utils.get_dataset(args.dataset, transform, args.train_split)
    
    # operate this train_loader to generate new loader
    
    train_loader = DataLoader(dataset['train_data'], batch_size = args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset['val_data'], batch_size = args.batch_size, shuffle=True)

    test_loader = DataLoader(dataset['test_data'], batch_size = args.batch_size, shuffle=False)

    image_size = 32
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    CE_loss = nn.CrossEntropyLoss()
    # jiang LeNet dan du chan fen le chu lai
    #model =  LeNet(output_classes)
    #model =  AlexNet(output_classes)

#   give the attacking model     single

    model = ResNet18(output_classes)

    #model = MoE_ResNet18(args.num_experts, output_classes)
    #model = MoE_ResNet18(9, output_classes)

    model = model.to(device)

    #model_loc = './trained_model/ckptnat_resnet18_cifar.pth'
    #model_loc = '/home/chenghao/Mixture_of_Experts-master/cifar_16_resenet/checkpoint/ckptMoE_resnet_cifar_clean+4adv+4nat.pth'

    ## single clean expert
    #model_loc = '../cifar_clean_train_resnet/checkpoint/ckptnat_resnet18_cifar_tsave.pth'
    model_loc = '../cifar_clean_train_resnet/checkpoint/ckptnat_resnet18_cifar_0.1_lu_l2_40*5_255.pth'


    ## all old
    #model_loc = '../cifar_16_resenet/checkpoint/ckptMoE_resnet_cifar_clean+4adv+4nat.pth'

    ## adv old

    #model_loc = '../cifar_16_resenet/checkpoint/ckptMoE_resnet_cifar_clean+4adv.pth'

    ## nat old

    #model_loc = '../cifar_16_resenet/checkpoint/ckptMoE_resnet_cifar_clean+4nat.pth'

    ## nat new

    #model_loc = '../cifar_16_resenet/checkpoint/ckptMoE_resnet_cifar_clean+4adv_true.pth'

    ## all new

    #model_loc = '../cifar_16_resenet/checkpoint/ckptMoE_resnet_cifar_clean+4adv+4nat_true.pth'

    #   give the attacking model MoE

    # model = MoE_ResNet18(args.num_experts, output_classes)
    #
    # model = model.to(device)
    # #model.cuda()
    #
    # model_loc = './trained_model/ckptMoE_resnet_cifar_clean+4adv.pth'



    utils.load_model(model_loc, model)


    # if device == 'cuda':
    #     model = torch.nn.DataParallel(model)
    #     cudnn.benchmark = True


    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    lr_schedule = lambda t: np.interp([t], [0, args.epochs*2//5, args.epochs*4//5, args.epochs], [0, 0.1, 0.005, 0])[0]

    if args.order == 'rand':  # frequency attack
        n_dims = 3 * args.freq_dims * args.freq_dims
    else:  # pixel attack
        n_dims = 3 * image_size * image_size  # cifar = 3 * 32 * 32 ;   imagenet 3 * 224 * 224

    # define the iteration number
    if args.num_iters > 0:
        max_iters = int(min(n_dims, args.num_iters))
        #max_iters = int(max(n_dims, args.num_iters))
    else:
        max_iters = int(n_dims)  # max_iters 可能是因为逐像素扫点
    #N = int(math.floor(float(args.num_runs) / float(args.batch_size)))  # define the epoch number

    model.eval()
    correct_final_nat = 0
    correct_final_l2 = 0
    correct_final_linf = 0

    acc_nat, best_acc_nat = val_clean(test_loader, device, model, correct_final_nat, best_acc_nat)

    print(' Done!!  maybe Natural  Accuracy: ', acc_nat)
    print('  Best Natural  Accuracy: ', best_acc_nat)

    adv, probs, succs, queries, l2_norms, linf_norms = val_simba(test_loader, model, args, max_iters)
    print('simba attack advs:', adv)
    print('simba attack probs:', probs)
    print('attack succss rate:', succs)
    print('num of queries:', queries)
    print('l2 norms:', l2_norms)
    print('linf norm', linf_norms)

    # for i in range(args.epochs):
    #     #model.train()
    #     #j = 0
    #     print('The epoch number is: ' + str(i))
    #
    #     lr = lr_schedule(i + (i+1)/args.batch_size)
    #
    #     #train_clean (train_loader, device,optimizer,model,CE_loss, lr_schedule, i)
    #
    #    # train_clean (train_loader, device,optimizer,model,CE_loss)
    #
    #    # train_clean (train_loader, device,optimizer,model,CE_loss)
    #
    #     #train_adv(train_loader, device, optimizer, basic_model, model, AttackPGD ,CE_loss,config_l2, attack)
    #
    #     #train_adv(train_loader, device, optimizer, basic_model, model, AttackPGD ,CE_loss,config_linf, attack)
    #
    #
    #
    #
    #
    #
    #
    #
    #     # acc_nat, best_acc_nat, acc_l2, best_acc_l2, acc_linf, best_acc_linf = val(test_loader, device, model,  model, AttackPGD, config_l2, config_linf, attack,\
    #     # correct_final_nat, best_acc_nat, correct_final_l2, best_acc_l2, correct_final_linf,\
    #     #     best_acc_linf, args.checkpoint_loc, max_iters)
    #
    #     acc_nat, best_acc_nat, probs, succs, queries, l2_norms, linf_norms = val_clean(test_loader, device, model, correct_final_nat, best_acc_nat, args, max_iters)
    #
    #     print('Epoch: ', i+1, ' Done!!  maybe Natural  Accuracy: ', acc_nat)
    #     print('Epoch: ', i+1, '  Best Natural  Accuracy: ', best_acc_nat)
    #
    #     print('simba attack probs:', probs)
    #     print('attack succss rate:', succs)
    #     print('num of queries:', queries)
    #     print('l2 norms:', l2_norms)
    #     print('linf norm', linf_norms)
    #
    #
        
        # #acc_2, best_acc_l2 = val_adv(val_loader, device, model,  model, AttackPGD, config_l2, attack, correct_final_2, best_acc_l2 ,args.checkpoint_loc)
        #
        # print('Epoch: ', i+1, ' Done!!  l2  Accuracy: ', acc_l2)
        # print('Epoch: ', i+1, '  Best l2  Accuracy: ', best_acc_l2)
        #
        # #acc_3, best_acc_linf = val_adv(val_loader, device, model,  model, AttackPGD, config_linf, attack, correct_final_3, best_acc_linf ,args.checkpoint_loc)
        #
        #
        #
        # print('Epoch: ', i+1, ' Done!!  l_inf  Accuracy: ', acc_linf)
        # print('Epoch: ', i+1, '  Best l_inf  Accuracy: ', best_acc_linf)
        #


def val_clean(loader, device, model, correct_final, best_acc):
    for images, labels in tqdm(loader):
        # images = images.to(device)

        # images_att = net_attack(images,labels, attack)

        images = images.to(device)
        labels = labels.to(device)

        prediction = model(images)
        pred = prediction.argmax(dim=1, keepdim=True)
        correct_1 = pred.eq(labels.view_as(pred)).sum().item()
        # correct_final.append(correct)
        correct_final = correct_final + correct_1

    acc = correct_final / len(loader.dataset)
    if acc > best_acc:
        print('updating..')
        best_acc = acc

    return acc, best_acc

def val_simba(loader, model, args, max_iters):
    i=0
    for images, labels in tqdm(loader):
        # images = images.to(device)

        # images_att = net_attack(images,labels, attack)
        adv, probs, succs, queries, l2_norms, linf_norms = dct_attack_batch(
            model, images, labels, max_iters, args.freq_dims, args.stride, args.epsilon,
            order=args.order,
            targeted=args.targeted, pixel_attack=args.pixel_attack, log_every=args.log_every)
        if i == 0:
            all_adv = adv
            all_probs = probs
            all_succs = succs
            all_queries = queries
            all_l2_norms = l2_norms
            all_linf_norms = linf_norms
        else:
            all_adv = torch.cat([all_adv, adv], dim=0)
            all_probs = torch.cat([all_probs, probs], dim=0)
            all_succs = torch.cat([all_succs, succs], dim=0)
            all_queries = torch.cat([all_queries, queries], dim=0)
            all_l2_norms = torch.cat([all_l2_norms, l2_norms], dim=0)
            all_linf_norms = torch.cat([all_linf_norms, linf_norms], dim=0)
        i = i+1
        # images = images.to(device)
        # labels = labels.to(device)
        #
        # prediction = model(images)
        # pred = prediction.argmax(dim=1, keepdim=True)
        # correct_1 = pred.eq(labels.view_as(pred)).sum().item()
        # # correct_final.append(correct)
        # correct_final = correct_final + correct_1

    # acc = correct_final / len(loader.dataset)
    # if acc > best_acc:
    #     print('updating..')
    #     best_acc = acc

    return all_adv, all_probs, all_succs, all_queries, all_l2_norms, all_linf_norms


# def val_simba(loader, device, model, correct_final, best_acc, args, max_iters):
#
#     for images, labels in tqdm(loader):
#
#
#             #images = images.to(device)
#
#             #images_att = net_attack(images,labels, attack)
#             images, probs, succs, queries, l2_norms, linf_norms = dct_attack_batch(
#                 model, images, labels, max_iters, args.freq_dims, args.stride, args.epsilon,
#                 order=args.order,
#                 targeted=args.targeted, pixel_attack=args.pixel_attack, log_every=args.log_every)
#
#             images = images.to(device)
#             labels = labels.to(device)
#
#             prediction = model(images)
#             pred = prediction.argmax(dim=1, keepdim=True)
#             correct_1 = pred.eq(labels.view_as(pred)).sum().item()
#             #correct_final.append(correct)
#             correct_final = correct_final + correct_1
#
#     acc = correct_final / len(loader.dataset)
#     if acc > best_acc:
#             print('updating..')
#             best_acc = acc
#
#     return acc, best_acc, probs, succs, queries, l2_norms, linf_norms

def val_adv(val_loader, device, model,  basic_model, AttackPGD, config, attack, correct_final, best_acc, checkpoint_loc):
    for images, labels in val_loader:
            
            #a = a+1
            net_attack = AttackPGD(basic_model,config)
            
            #print(net_attack)
            
            net_attack = net_attack.to(device)
            #print('processing testing image:', a)
            #images, labels = utils.cuda([images, labels], args.gpu_ids)
            images = images.to(device)
            labels = labels.to(device)
            
            images_att = net_attack(images,labels, attack)

            prediction = model(images_att)
            pred = prediction.argmax(dim=1, keepdim=True)
            correct_2 = pred.eq(labels.view_as(pred)).sum().item()
            #correct_final.append(correct)
            correct_final = correct_final + correct_2
        
    acc = correct_final / len(val_loader.dataset)
    
    if acc > best_acc:
            print('saving..')
            
            state = {
            'net': model.state_dict(),
            'acc': acc,    #zhu ming type
            #'epoch': epoch,
            }

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, checkpoint_loc)
            best_acc = acc
    
    return acc, best_acc





