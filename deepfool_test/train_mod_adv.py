#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 15:54:37 2020

@author: chad
"""

import os
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

import utils
# from model_adv import AlexNet, MoE_alexnet
# from model_adv_att import AttackPGD
from model_deepfool_att import deepfool
from model_resnet import *
import string




def testing(val_loader, device, model, basic_model, AttackPGD, config_l2, config_linf, attack, \
        correct_final_nat, best_acc_nat, correct_final_l2, best_acc_l2, correct_final_linf, \
        best_acc_linf, best_acc_aver, checkpoint_loc):
    acc_linf = []
    acc_l2 = []

    print('Valuation clean images')

    for images_1, labels in tqdm(val_loader):
        images_1 = images_1.to(device)
        labels = labels.to(device)

        # images_att = net_attack(images,labels, attack)

        prediction, weights = model(images_1)
        pred = prediction.argmax(dim=1, keepdim=True)
        correct_1 = pred.eq(labels.view_as(pred)).sum().item()
        # correct_final.append(correct)
        correct_final_nat = correct_final_nat + correct_1

    acc_nat = correct_final_nat / len(val_loader.dataset)

    j = 0

    for i in config_l2:
        j = j + 1
        # print('Valuation l_2  epsilon: ' + str(config_l2[i]['epsilon']))
        print('Valuation l_2  epsilon: 50+ ' + str(j) + '*10 / 255')

        correct_2 = 0
        correct_final_l2 = 0

        for images_2, labels in tqdm(val_loader):
            # a = a+1
            net_attack = AttackPGD(basic_model, config_l2[i])

            # print(net_attack)

            net_attack = net_attack.to(device)
            # print('processing testing image:', a)
            # images, labels = utils.cuda([images, labels], args.gpu_ids)
            images_2 = images_2.to(device)
            labels = labels.to(device)

            images_att = net_attack(images_2, labels, attack)

            prediction, weights = model(images_att)
            pred = prediction.argmax(dim=1, keepdim=True)
            correct_2 = pred.eq(labels.view_as(pred)).sum().item()
            # correct_final.append(correct)
            correct_final_l2 = correct_final_l2 + correct_2

        acc_l2.append(correct_final_l2 / len(val_loader.dataset))

    j = 0
    for i in config_linf:

        j = j + 1

        # print('Valuation' + str(config_linf[i]['_type']) + '  epsilon:' + str(config_linf[i]['epsilon']))
        print('Valuation l_inf  epsilon: 5+ ' + str(j) + '*10 / 255')
        correct_2 = 0
        correct_final_linf = 0
        for images_3, labels in tqdm(val_loader):
            # a = a+1
            net_attack = AttackPGD(basic_model, config_linf[i])

            # print(net_attack)

            net_attack = net_attack.to(device)
            # print('processing testing image:', a)
            # images, labels = utils.cuda([images, labels], args.gpu_ids)
            images_3 = images_3.to(device)
            labels = labels.to(device)

            images_att = net_attack(images_3, labels, attack)

            prediction, weights = model(images_att)
            pred = prediction.argmax(dim=1, keepdim=True)
            correct_3 = pred.eq(labels.view_as(pred)).sum().item()
            # correct_final.append(correct)
            correct_final_linf = correct_final_linf + correct_3

        acc_linf.append(correct_final_linf / len(val_loader.dataset))

    if (acc_nat * 2 + sum(acc_l2) + sum(acc_linf)) / 6 > best_acc_aver:
        print('updating..')
        best_acc_nat = acc_nat
        best_acc_l2 = acc_l2
        best_acc_linf = acc_linf
        best_acc_aver = (best_acc_nat * 2 + sum(best_acc_l2) + sum(best_acc_linf)) / 6

    return acc_nat, best_acc_nat, acc_l2, best_acc_l2, acc_linf, best_acc_linf, best_acc_aver


def testing_deepfool(val_loader, device, model, basic_model, args):

    acc = []

    j = 0

    # for i in config_l2:
    #     j = j + 1
        # print('Valuation l_2  epsilon: ' + str(config_l2[i]['epsilon']))
    print('Valuation deepfool')

    correct = 0
    correct_final = 0

    for images_2, labels in tqdm(val_loader):
        # a = a+1
        _, _, _, _, pert_image = deepfool(images_2, basic_model, norm=args.norm, eps=args.epsilon)

        # print(net_attack)

        # net_attack = net_attack.to(device)
        # print('processing testing image:', a)
        # images, labels = utils.cuda([images, labels], args.gpu_ids)
        pert_image = pert_image.to(device)
        labels = labels.to(device)

        # images_att = net_attack(pert_image, labels, attack)

        prediction = model(pert_image)
        pred = prediction.argmax(dim=1, keepdim=True)
        correct = pred.eq(labels.view_as(pred)).sum().item()
        # correct_final.append(correct)
        correct_final = correct_final + correct

    acc = correct_final / len(val_loader.dataset)



    # if (acc_nat * 2 + sum(acc_l2) + sum(acc_linf)) / 6 > best_acc_aver:
    #     print('updating..')
    #     best_acc_nat = acc_nat
    #     best_acc_l2 = acc_l2
    #     best_acc_linf = acc_linf
    #     best_acc_aver = (best_acc_nat * 2 + sum(best_acc_l2) + sum(best_acc_linf)) / 6

    return acc


def test(args):

    config_linf_6 = {
        'epsilon': 6.0 / 255,
        'num_steps': 7,
        # 'step_size': 2.0 / 255,
        'step_size': 0.01,
        'random_start': True,
        'loss_func': 'xent',
        '_type': 'linf'
    }
    config_linf_8 = {
        'epsilon': 8.0 / 255,
        'num_steps': 7,
        # 'step_size': 2.0 / 255,
        'step_size': 0.01,
        'random_start': True,
        'loss_func': 'xent',
        '_type': 'linf'
    }

    config_linf = dict(config_linf_6=config_linf_6, config_linf_8=config_linf_8)


    config_l2_1_2 = {
        'epsilon': 0.5,
        'num_steps': 7,
        # 'step_size': 2.0 / 255,
        'step_size': 0.5 / 5,
        'random_start': True,
        'loss_func': 'xent',
        '_type': 'l2'
    }
    config_l2_1 = {
        'epsilon': 1.0,
        'num_steps': 7,
        # 'step_size': 2.0 / 255,
        'step_size': 1.0 / 5,
        'random_start': True,
        'loss_func': 'xent',
        '_type': 'l2'
    }


    config_l2 = dict(config_l2_1_2=config_l2_1_2, config_l2_60=config_l2_1)
    attack = 'true'

    if args.dataset == 'cifar':
        output_classes = 10

    global best_acc_nat, best_acc_l2, best_acc_linf
    best_acc_nat = 0
    best_acc_l2 = []
    best_acc_linf = []
    best_acc_aver = 0

    transform = utils.get_transformation(args.dataset)
    dataset = utils.get_dataset(args.dataset, transform, args.train_split)

    # operate this train_loader to generate new loader

    test_loader = DataLoader(dataset['test_data'], batch_size=args.batch_size, shuffle=True)

    # l2 attack and linf attack to generate new data

    # for images, labels in train_loader:
    #     print(images)
    #     print(images.type)
    #     print(images.shape)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    CE_loss = nn.CrossEntropyLoss()
    # jiang LeNet dan du chan fen le chu lai
    # model =  LeNet(output_classes)
    basic_model = ResNet18(output_classes)
    basic_model = basic_model.to(device)
    # print(output_classes.device)
    model = MoE_ResNet18(args.num_experts, output_classes)
    # model_adv = MoE_ResNet18_adv(args.num_experts, output_classes)
    # model_adv = ResNet18(output_classes)

    model = model.to(device)
    utils.load_model(args.checkpoint_loc, model)




    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    # model = utils.cuda(model, args.gpu_ids)
    # model.cuda(args.gpu_ids[0])
    # model = model.to(device)
    # model = torch.nn.DataParallel(model, device_ids = [0]).cuda()
    # print(model.device)
    # model = MoE(basic_model, args.num_experts, output_classes)
    # print(model)
    # print(type(model))
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_schedule = lambda t: \
    np.interp([t], [0, args.epochs * 2 // 5, args.epochs * 4 // 5, args.epochs], [0, 0.1, 0.005, 0])[0]

    for i in range(args.epochs):
        # model.train()
        # j = 0
        print('The epoch number is: ' + str(i))

        model.eval()
        correct_final_nat = 0
        correct_final_l2 = 0
        correct_final_linf = 0

        acc_nat, best_acc_nat, acc_l2, best_acc_l2, acc_linf, best_acc_linf, best_acc_aver = testing(test_loader, device,
                                                                                                 model, model,
                                                                                                 AttackPGD, config_l2,
                                                                                                 config_linf, attack, \
                                                                                                 correct_final_nat,
                                                                                                 best_acc_nat,
                                                                                                 correct_final_l2,
                                                                                                 best_acc_l2,
                                                                                                 correct_final_linf, \
                                                                                                 best_acc_linf,
                                                                                                 best_acc_aver,
                                                                                                 args.checkpoint_loc)

        # acc_1, best_acc_nat = val_clean(val_loader, device, model, correct_final_1, best_acc_nat, args.checkpoint_loc)

        print('Epoch: ', i + 1, ' Done!!  Natural  Accuracy: ', acc_nat)
        print('Epoch: ', i + 1, '  Best Natural  Accuracy: ', best_acc_nat)

        # acc_2, best_acc_l2 = val_adv(val_loader, device, model,  model, AttackPGD, config_l2, attack, correct_final_2, best_acc_l2 ,args.checkpoint_loc)

        print('Epoch: ', i + 1, ' Done!!  l2(50, ..., 110)  Accuracy: ', acc_l2)
        print('Epoch: ', i + 1, '  Best l2  Accuracy: ', best_acc_l2)

        # acc_3, best_acc_linf = val_adv(val_loader, device, model,  basic_model, AttackPGD, config_linf, attack, correct_final_3, best_acc_linf ,args.checkpoint_loc)

        print('Epoch: ', i + 1, ' Done!!  l_inf(5, ..., 11)  Accuracy: ', acc_linf)
        print('Epoch: ', i + 1, '  Best l_inf  Accuracy: ', best_acc_linf)

        # print('Epoch: ', i+1, ' Done!!    Loss: ', loss)

        print('Epoch: ', i + 1, ' Done!!  average Accuracy: ', (acc_nat * 2 + sum(acc_l2) + sum(acc_linf)) / 6)
        print('Epoch: ', i + 1, '  Best average  Accuracy: ', best_acc_aver)

def test_deepfool(args):

    # attack = 'true'

    if args.dataset == 'cifar':
        output_classes = 10

    global best_acc_nat, best_acc_l2, best_acc_linf
    best_acc_nat = 0
    best_acc_l2 = []
    best_acc_linf = []
    best_acc_aver = 0

    transform = utils.get_transformation(args.dataset)
    dataset = utils.get_dataset(args.dataset, transform, args.train_split)

    # operate this train_loader to generate new loader

    test_loader = DataLoader(dataset['test_data'], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset['val_data'], batch_size=args.batch_size, shuffle=True)

    # l2 attack and linf attack to generate new data

    # for images, labels in train_loader:
    #     print(images)
    #     print(images.type)
    #     print(images.shape)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    CE_loss = nn.CrossEntropyLoss()
    # jiang LeNet dan du chan fen le chu lai
    # model =  LeNet(output_classes)
    # model = ResNet18(output_classes)
    # basic_model = basic_model.to(device)
    # print(output_classes.device)
    model = MoE_ResNet18_adv(args.num_experts, output_classes)
    # model_adv = MoE_ResNet18_adv(args.num_experts, output_classes)
    # model_adv = ResNet18(output_classes)

    # model_loc = '../cifar_clean_train_resnet/checkpoint/ckpt_resnet18_cifar_clean.pth'
    # model_loc = '../cifar_clean_train_resnet/checkpoint/l2_1.0.pt'

    model = model.to(device)
    utils.load_model(args.checkpoint_loc, model)
    # utils.load_model(model_loc, model)





    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    # model = utils.cuda(model, args.gpu_ids)
    # model.cuda(args.gpu_ids[0])
    # model = model.to(device)
    # model = torch.nn.DataParallel(model, device_ids = [0]).cuda()
    # print(model.device)
    # model = MoE(basic_model, args.num_experts, output_classes)
    # print(model)
    # print(type(model))
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_schedule = lambda t: \
    np.interp([t], [0, args.epochs * 2 // 5, args.epochs * 4 // 5, args.epochs], [0, 0.1, 0.005, 0])[0]

    for i in range(args.epochs):
        # model.train()
        # j = 0
        print('The epoch number is: ' + str(i))

        model.eval()
        correct_final_nat = 0
        correct_final_l2 = 0
        correct_final_linf = 0

        acc_deepfool = testing_deepfool(val_loader, device, model, model, args)

        # acc_1, best_acc_nat = val_clean(val_loader, device, model, correct_final_1, best_acc_nat, args.checkpoint_loc)

        print('Epoch: ', i + 1, ' Done!!  deepfool  Accuracy: ', acc_deepfool)

