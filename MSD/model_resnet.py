#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 20:47:06 2020

@author: chad
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNet_expert(nn.Module):
    def __init__(self, block, num_blocks, output_dim):
        super(ResNet_expert, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        for p in self.parameters():    # fc  could train
            p.requires_grad=False
        self.linear = nn.Linear(512*block.expansion, output_dim)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(output_dim):
    return ResNet(BasicBlock, [2,2,2,2], output_dim)

def ResNet18_expert(output_dim):
    return ResNet_expert(BasicBlock, [2,2,2,2], output_dim)

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

class MoE_ResNet18(nn.Module):
    def __init__(self, num_experts, output_dim):
        super(MoE_ResNet18, self).__init__()
        self.num_experts = num_experts
        self.output_dim = output_dim
        self.softmax = nn.Softmax()

        self.experts = nn.ModuleList([ResNet18_expert(output_dim) for i in range(num_experts)])

        self.gating = ResNet18(num_experts)

    def forward(self, x):

        out_final = []
        weights = self.softmax(self.gating(x))    ### Outputs a tensor of [batch_size, num_experts]
        # print(x)
        # print(len(x))
        # print(x.shape)
        # print(weights)
        # print(len(weights))
        # print(weights.shape)
        out = torch.zeros([weights.shape[0], self.output_dim])
        for i in range(self.num_experts):
            #out += weights[:, i].unsqueeze(1) * self.experts[i](x)    ### To get the output of experts weighted by the appropriate gating weight
            out = weights[:, i].unsqueeze(1) * self.experts[i](x)

            # print('out is :', out)

            out_final.append(out)

            # print('all out are:' , out_final)

            # print('size of out:', len(out_final))

        weights_aver = torch.mean(weights, dim=0, keepdim=True)
        return sum(out_final), weights_aver    ### out will have the shape [batch_size, output_dim]
        #return sum(out_final)    ### out will have the shape [batch_size, output_dim]


class MoE_ResNet18_test(nn.Module):
    def __init__(self, num_experts, output_dim):
        super(MoE_ResNet18_test, self).__init__()
        self.num_experts = num_experts
        self.output_dim = output_dim
        self.softmax = nn.Softmax()

        self.experts = nn.ModuleList([ResNet18(output_dim) for i in range(num_experts)])

        self.gating = ResNet18(num_experts)

    def forward(self, x):
        out_final = []
        weights = self.softmax(self.gating(x))  ### Outputs a tensor of [batch_size, num_experts]
        # print(x)
        # print(len(x))
        # print(x.shape)
        # print(weights)
        # print(len(weights))
        # print(weights.shape)
        out = torch.zeros([weights.shape[0], self.output_dim])
        for i in range(self.num_experts):
            # out += weights[:, i].unsqueeze(1) * self.experts[i](x)    ### To get the output of experts weighted by the appropriate gating weight
            out = weights[:, i].unsqueeze(1) * self.experts[i](x)

            # print('out is :', out)

            out_final.append(out)

            # print('all out are:' , out_final)

            # print('size of out:', len(out_final))

        #weights_aver = torch.mean(weights, dim=0, keepdim=True)
        return sum(out_final)  ### out will have the shape [batch_size, output_dim]
        # return sum(out_final)    ### out will have the shape [batch_size, output_dim]

class MoE_ResNet18_adv(nn.Module):
    def __init__(self, num_experts, output_dim):
        super(MoE_ResNet18_adv, self).__init__()
        self.num_experts = num_experts
        self.output_dim = output_dim
        self.softmax = nn.Softmax()

        self.experts = nn.ModuleList([ResNet18_expert(output_dim) for i in range(num_experts)])

        self.gating = ResNet18(num_experts)

    def forward(self, x):
        out_final = []
        weights = self.softmax(self.gating(x))  ### Outputs a tensor of [batch_size, num_experts]
        # print(x)
        # print(len(x))
        # print(x.shape)
        # print(weights)
        # print(len(weights))
        # print(weights.shape)
        out = torch.zeros([weights.shape[0], self.output_dim])
        for i in range(self.num_experts):
            # out += weights[:, i].unsqueeze(1) * self.experts[i](x)    ### To get the output of experts weighted by the appropriate gating weight
            out = weights[:, i].unsqueeze(1) * self.experts[i](x)

            # print('out is :', out)

            out_final.append(out)

            # print('all out are:' , out_final)

            # print('size of out:', len(out_final))

        #weights_aver = torch.mean(weights, dim=0, keepdim=True)
        return sum(out_final)  ### out will have the shape [batch_size, output_dim]
        # return sum(out_final)    ### out will have the shape [batch_size, output_dim]


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
