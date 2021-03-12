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
from model_adv_att import AttackPGD
from model_resnet import *
import string


def train_clean (train_loader, device,optimizer,model,CE_loss, lr_schedule, epoch_i):
    a = 0
    for images, labels in tqdm(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            # optimizer.zero_grad()
            prediction = model(images)
            
            #print('prediction value is :', prediction )

            loss = CE_loss(prediction, labels)
            # print('loss value is :', loss )
            
            lr = lr_schedule(epoch_i + (a+1)/len(train_loader))
            optimizer.param_groups[0].update(lr=lr)

            optimizer.zero_grad()
            
            loss.backward()

            optimizer.step()
            
            a = a+1

def train_adv(train_loader, device,optimizer, basic_model, model, AttackPGD ,CE_loss,config,attack, lr_schedule, epoch_i):
    print('doing l1 training')
    b = 0
    for images, labels in tqdm(train_loader):
            model.eval()
            basic_model.eval()
            net_attack = AttackPGD(basic_model,config)
            
            #print(net_attack)
            
            net_attack = net_attack.to(device)
            images = images.to(device)
            labels = labels.to(device)
            #print(images.device)
            # images.cuda(args.gpu_ids[0])
            # labels.cuda(args.gpu_ids[0])
            #print(images)
            #print(images.shape)
            #print(type(images))
            
            images_att = net_attack(images,labels, attack)
            

            model.train()
            basic_model.train()
            #optimizer.zero_grad()
            prediction = model(images_att)
            
            #print('prediction value is :', prediction )

            loss = CE_loss(prediction, labels)

            lr = lr_schedule(epoch_i + (b+1)/len(train_loader))
            optimizer.param_groups[0].update(lr=lr)

            optimizer.zero_grad()
            # print('loss value is :', loss )
            loss.backward()

            optimizer.step()
            
            b = b+1



def val(val_loader, device, model,  basic_model, AttackPGD, config_l1, config_l2, config_linf, attack,\
        correct_final_nat, best_acc_nat, correct_final_l1, best_acc_l1, correct_final_linf, best_acc_linf, checkpoint_loc):
    
    acc_nat = 0
    acc_l1 = 0
    acc_linf =0


    for images, labels in tqdm(val_loader):
            
 
            images = images.to(device)
            labels = labels.to(device)
            
            #images_att = net_attack(images,labels, attack)
    
            prediction = model(images)
            pred = prediction.argmax(dim=1, keepdim=True)
            correct_1 = pred.eq(labels.view_as(pred)).sum().item()
            #correct_final.append(correct)
            correct_final_nat = correct_final_nat + correct_1
            
    acc_nat = correct_final_nat / len(val_loader.dataset)
    
    for images, labels in tqdm(val_loader):
            
            #a = a+1
            net_attack = AttackPGD(basic_model,config_l1)
            
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
            correct_final_l1 = correct_final_l1 + correct_2
        
    acc_l1 = correct_final_l1 / len(val_loader.dataset)


    
    
    
    

    if (acc_l1+acc_nat)/2.0 > best_acc_l1:
            print('saving..')
            
            state = {
            'net': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
            'acc_clean': acc_nat,
            'acc_l1': acc_l1,
            'acc_linf': acc_linf,
            #'epoch': epoch,
            }

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, checkpoint_loc)
            best_acc_nat = acc_nat
            best_acc_l1 = (acc_l1+acc_nat)/2.0
            best_acc_linf = acc_linf
    
    return acc_nat, best_acc_nat, acc_l1, best_acc_l1, acc_linf, best_acc_linf



def train(args):
    
    # config_linf = {
    # 'epsilon': 4.0  / 255 ,
    # #'epsilon': 0.314,
    # 'num_steps': 10,
    # 'step_size': 2.0 / 255,
    # 'random_start': True,
    # 'loss_func': 'xent',
    # '_type': 'linf'
    #  }
    
    # config_l2 = {
    # 'epsilon': 40  / 255,
    # #'epsilon': 0.314 * 5,
    # 'num_steps': 10,
    # 'step_size': 2.0 / 255,
    # 'random_start': True,
    # 'loss_func': 'xent',
    # '_type': 'l2'
    #  }

    #region setting
    if args.heavy == 1: 
        config_linf = {
        'epsilon': 8.0  / 255 ,
        'num_steps': 7,
        'step_size': 0.01,
        'random_start': True,
        'loss_func': 'xent',
        '_type': 'linf'
        }

        config_l2 = {
        'epsilon': 1.0,
        'num_steps': 7,
        'step_size': 1.0/5,
        'random_start': True,
        'loss_func': 'xent',
        '_type': 'l2'
        }

        config_l1 = {
        'epsilon': 16.0,
        'num_steps': 7,
        'step_size': 2.5 * 16.0 / 7,
        'random_start': True,
        'loss_func': 'xent',
        '_type': 'l1'
        }
    else:
        config_linf = {
        'epsilon': 6.0  / 255 ,
        'num_steps': 7,
        'step_size': 0.01,
        'random_start': True,
        'loss_func': 'xent',
        '_type': 'linf'
        }
        config_l2 = {
        'epsilon': 0.5,
        'num_steps': 7,
        'step_size': 0.5/5,
        'random_start': True,
        'loss_func': 'xent',
        '_type': 'l2'
        }
        config_l1 = {
        'epsilon': 12.0,
        'num_steps': 7,
        'step_size': 2.5 * 12.0 / 7,
        'random_start': True,
        'loss_func': 'xent',
        '_type': 'l1'
        }
    import os
    dir = args.out_dir
    if not os.path.exists(dir):
        os.makedirs(dir)
    args.checkpoint_loc = '{}{}_{}_{}.pt'.format(dir, args.dataset,'l1',config_l1['epsilon'])
    print(args.checkpoint_loc)
    #endregion


    attack = 'true'

    if args.dataset == 'cifar':
        output_classes = 10
    elif args.dataset == 'tinyimagenet':
        output_classes = 200
        
    global best_acc_nat, best_acc_l1, best_acc_linf
    best_acc_nat = 0
    best_acc_l1 = 0
    best_acc_linf = 0
    

    transform = utils.get_transformation(args.dataset)
    dataset = utils.get_dataset(args.dataset, transform, args.train_split)
    
    # operate this train_loader to generate new loader
    
    train_loader = DataLoader(dataset['train_data'], batch_size = args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset['test_data'], batch_size = args.batch_size, shuffle=True)
    

    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    CE_loss = nn.CrossEntropyLoss()
    # jiang LeNet dan du chan fen le chu lai
    #model =  LeNet(output_classes)
    #model =  AlexNet(output_classes)
    model = ResNet18(output_classes)

    model = model.to(device)
    if args.resume:
        utils.load_model(args.checkpoint_loc, model)
        print("loaded model!")

    if device == 'cuda':
        # model = torch.nn.DataParallel(model)
        cudnn.benchmark = True


    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    lr_schedule = lambda t: np.interp([t], [0, args.epochs*2//5, args.epochs*4//5, args.epochs], [0, 0.1, 0.005, 0])[0]

    for i in range(args.epochs):
        model.train()
        #j = 0
        print('The epoch number is: ' + str(i))
        
        lr = lr_schedule(i + (i+1)/args.batch_size)
        
        #train_clean (train_loader, device,optimizer,model,CE_loss, lr_schedule, i)
        
       # train_clean (train_loader, device,optimizer,model,CE_loss)
        
       # train_clean (train_loader, device,optimizer,model,CE_loss)
        
        train_adv(train_loader, device, optimizer, model, model, AttackPGD ,CE_loss,config_l1, attack, lr_schedule, i)
            
        #train_adv(train_loader, device, optimizer, basic_model, model, AttackPGD ,CE_loss,config_linf, attack)
        
        


        model.eval()
        correct_final_nat = 0
        correct_final_l1 = 0
        correct_final_linf = 0

        
        
        acc_nat, best_acc_nat, acc_l1, best_acc_l1, acc_linf, best_acc_linf = val(test_loader, device, model,  model, AttackPGD,config_l1, config_l2, config_linf, attack,\
        correct_final_nat, best_acc_nat, correct_final_l1, best_acc_l1, correct_final_linf,\
            best_acc_linf, args.checkpoint_loc)
        
        # acc_nat, best_acc_nat = val_clean(val_loader, device, model, correct_final_nat, best_acc_nat, args.checkpoint_loc)
        
        print('Epoch: ', i+1, ' Done!!  Natural  Accuracy: ', acc_nat)
        print('Epoch: ', i+1, '  Best Natural  Accuracy: ', best_acc_nat) 
          
        
        
        
        #acc_2, best_acc_l2 = val_adv(val_loader, device, model,  model, AttackPGD, config_l2, attack, correct_final_2, best_acc_l2 ,args.checkpoint_loc)
        
        print('Epoch: ', i+1, ' Done!!  l1  Accuracy: ', acc_l1)
        print('Epoch: ', i+1, '  Best l1  Accuracy: ', best_acc_l1)         
        
        #acc_3, best_acc_linf = val_adv(val_loader, device, model,  model, AttackPGD, config_linf, attack, correct_final_3, best_acc_linf ,args.checkpoint_loc)


        
        print('Epoch: ', i+1, ' Done!!  l_inf  Accuracy: ', acc_linf)
        print('Epoch: ', i+1, '  Best l_inf  Accuracy: ', best_acc_linf)
        



    

            

















def val_clean(val_loader, device, model, correct_final, best_acc, checkpoint_loc):
    
    for images, labels in val_loader:
            
 
            images = images.to(device)
            labels = labels.to(device)
            
            #images_att = net_attack(images,labels, attack)
    
            prediction = model(images)
            pred = prediction.argmax(dim=1, keepdim=True)
            correct_1 = pred.eq(labels.view_as(pred)).sum().item()
            #correct_final.append(correct)
            correct_final = correct_final + correct_1
            
    acc = correct_final / len(val_loader.dataset)
    if acc > best_acc:
            print('saving..')
            
            state = {
            'net': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
            'acc': acc,
            #'epoch': epoch,
            }

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, checkpoint_loc)
            best_acc = acc
    
    return acc, best_acc

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
            'net': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
            'acc': acc,    #zhu ming type
            #'epoch': epoch,
            }

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, checkpoint_loc)
            best_acc = acc
    
    return acc, best_acc
