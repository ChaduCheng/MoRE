import os
from argparse import ArgumentParser
import torch
import utils
from model_resnet import ResNet18
import numpy as np
from tqdm import tqdm
from art.attacks.evasion import HopSkipJump
from art.estimators.classification import PyTorchClassifier
from torch.utils.data import random_split
#from testing import validation
# from train_mod_lu import train
# from tests import test


def get_args():
    parser = ArgumentParser(description='Mixture of Experts')
    # parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='tinyimagenet')
    # parser.add_argument('--batch_size', type=int, default=8)
    # parser.add_argument('--train_split', type=float, default=0.8)
    # parser.add_argument('--lr', type=float, default=.001)
    # parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--checkpoint_loc', type=str, default='./ckpt_resnet18/clean.pth')
    # parser.add_argument('--num_experts', type=int, default=16)
    # parser.add_argument('--training', type=bool, default=True)
    # parser.add_argument('--testing', type=bool, default=False)
    args = parser.parse_args()
    args.resume = True
    args.epochs = 200
    args.batch_size = 128
    args.train_split = 0.8 
    args.lr = 0.1
    args.gpu_ids = '0'
    # args.checkpoint_loc = './ckpt_resnet18/clean.pth'
    args.num_experts = 3
    args.training = False
    args.testing = True
    return args
def test(args):
    if args.dataset == 'cifar':
        output_classes = 10
    if args.dataset == 'tinyimagenet':
        output_classes = 200
    transform = utils.get_transformation(args.dataset)
    dataset = utils.get_dataset(args.dataset, transform, args.train_split)
    train_loader = torch.utils.data.DataLoader(dataset['train_data'], batch_size = args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset['test_data'], batch_size = args.batch_size, shuffle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    CE_loss = torch.nn.CrossEntropyLoss()
    model = ResNet18(output_classes)
    model = model.to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.checkpoint_loc)['net'])
    else: 
        raise Exception("not give a model to load!")
    if device == 'cuda':
        # model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    lr_schedule = lambda t: np.interp([t], [0, args.epochs*2//5, args.epochs*4//5, args.epochs], [0, 0.1, 0.005, 0])[0]
    min_pixel_value = 0.0
    max_pixel_value = 1.0
    classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=CE_loss,
    optimizer=optimizer,
    input_shape=(3, 64, 64),
    nb_classes=output_classes,
    )
    test_clean(test_loader, device, classifier, model)
    norm = 2
    test_adv(test_loader, device, norm,classifier, model)
    norm = "inf"
    test_adv(test_loader, device, norm,classifier, model)


def test_adv(test_loader, device, norm,classifier, model):
    total = 0
    correct = 0
    attack= HopSkipJump(classifier=classifier,targeted=False, norm=norm, max_iter=10, max_eval=1, init_eval=1, init_size =1, verbose=True)
    with torch.no_grad():
        for idx, (images, labels) in tqdm(enumerate(test_loader)):
            images_adv = attack.generate(x=images)
            images , labels = images.to(device), labels.to(device)
            images_adv = torch.from_numpy(images_adv).to(device)
            outputs = model(images_adv)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # break
    print('Accuracy of the network on the norm '+ str(norm) +' test images: %d %%' % (
        100 * correct / total))
def test_clean(test_loader, device, classifier,model):
    total = 0
    correct = 0
    with torch.no_grad():
        for idx, (images, labels) in tqdm(enumerate(test_loader)):
            images , labels = images.to(device), labels.to(device)  
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the benign test images: %d %%' % (
        100 * correct / total))
                
                
def main():
    args = get_args()
    # str_ids = args.gpu_ids.split(',')
    # args.gpu_ids = []
    # for str_id in str_ids:
    #     id = int(str_id)
    #     if id >= 0:
    #         args.gpu_ids.append(id)
    
    # if args.training:
    #     train(args)
    if args.testing:
        test(args)


if __name__ == '__main__':
    main()