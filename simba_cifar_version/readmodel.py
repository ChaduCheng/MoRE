'''Read models.'''
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from model_resnet import *
from normal import *


def readmodel(name, device):
    if name == 'resnet18':
        net = ResNet18(10)
    elif name == 'moe':
        net = MoE_ResNet18(5, 10)


    #net = nn.Sequential(NormalizeLayer(), net)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
    return net

def loadmodel(net, name):
    ckpname = ('./checkpoint/' + '_' + name + '.pth')
    checkpoint = torch.load(ckpname)
    net.load_state_dict(checkpoint['net'])


def loaddivmodel(net, name, filename):
    ckpname = ('./checkpoint/' + filename + '.pth')
    checkpoint = torch.load(ckpname)
    #net.load_state_dict(checkpoint[name])
    net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['net'].items()})


def load_model(model_path, basic_net):
    checkpoint = torch.load(model_path)
    
    #print(checkpoint)
    
    #basic_net.load_state_dict(checkpoint['net'])
    basic_net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['net'].items()})
    # best_acc_nat = checkpoint['acc_clean']
    # best_acc_l2 = checkpoint['acc_l2']
    # best_acc_linf = checkpoint['acc_linf']
    
    # return best_acc_nat, best_acc_l2, best_acc_linf
