import argparse
import os
from PIL import Image
import torch
import torchvision
from torchvision import datasets
import utils
from tqdm import tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default = True, action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--batch_size', '-bs',type=int,  default =2,help='batch size')
parser.add_argument("--model_path", "-p", type=str, default=None)
args = parser.parse_args()
def l2_project(x,orig_input,eps):
    """
    """
    diff = x - orig_input
    diff = diff.renorm(p=2, dim=0, maxnorm=eps)
    return torch.clamp(orig_input + diff, 0, 1)
def linf_project(x,orig_input,eps):
    """
    """
    diff = x - orig_input
    diff = torch.clamp(diff, -eps, eps)
    return torch.clamp(diff + orig_input, 0, 1)
def normalize(x):
    return utils.apply_normalization(x, 'cifar')

def get_probs(model, x, y):
    model=model.to(device)
    # output = model(normalize(x.to(device))).to(device)
    output = model(normalize(x))
    probs = torch.nn.Softmax(dim=1)(output)[:, y]
    return torch.diag(probs.data)

# 20-line implementation of (untargeted) SimBA for single image input
def simba_single(model, x, y, num_iters=10000, epsilon=0.2):
    n_dims = x.view(1, -1).size(1)
    perm = torch.randperm(n_dims)
    last_prob = get_probs(model, x, y)
    for i in tqdm(range(num_iters)):
        diff = torch.zeros(n_dims)
        diff[perm[i]] = epsilon
        left_prob = get_probs(model, (x - diff.view(x.size()).to(device)).clamp(0, 1), y)
        if left_prob < last_prob:
            x = (x - diff.view(x.size()).to(device)).clamp(0, 1)
            last_prob = left_prob
        else:
            right_prob = get_probs(model, (x + diff.view(x.size()).to(device)).clamp(0, 1), y)
            if right_prob < last_prob:
                x = (x + diff.view(x.size()).to(device)).clamp(0, 1)
                last_prob = right_prob
    return x


def test(net,testloader):
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=args.lr,
    #                     momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print( 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=utils.CIFAR_TRANSFORM)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=utils.CIFAR_TRANSFORM)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=True, num_workers=2)
from resnet import ResNet18
net = ResNet18()
if device == 'cuda':
    torch.backends.cudnn.benchmark = True
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    # checkpoint = torch.load('../single_expert_training/checkpoint/ckptnat_resnet18_cifar_0.1_lu_l2_40_255.pth')
    # path = './checkpoint/ckpt.pth'
    # path = './checkpoint/ckptnat_resnet18_cifar_0.1_lu_linf_6_255.pth'
    # path = './checkpoint/ckptnat_resnet18_cifar_tsave.pth'
    path = args.model_path
    # net = torch.nn.DataParallel(net)
    # checkpoint = torch.load(path)
    # net.load_state_dict(checkpoint['net'])
    # torch.save({'net':net.module.state_dict()}, path)
    # quit()
    
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['net'])
    


    # best_acc = checkpoint['acc']
    # start_epoch = checkpoint['epoch']
net = net.to(device)


##################################
class adv_cifar(datasets.CIFAR10):
  def __init__(self, **kwargs):
    # super(adv_cifar, self).__init__(*kwargs)
    super().__init__(**kwargs)

  def __getitem__(self, index):
    img, target = self.data[index], self.targets[index]
    img = Image.fromarray(img)
    if self.transform is not None:
        img = self.transform(img)
    if self.target_transform is not None:
        target = self.target_transform(target)
    simba_single(model=net, x=img, y=target, num_iters=10, epsilon=0.4)
    return img, target
# trainset2 =adv_cifar(root='./data', train=True, download=True, transform=utils.CIFAR_TRANSFORM)
# trainloader2 = torch.utils.data.DataLoader(
#     trainset2, batch_size=args.batch_size, shuffle=True, num_workers=2)
# testset2 = adv_cifar(
#     root='./data', train=False, download=True, transform=utils.CIFAR_TRANSFORM)
# testloader2 = torch.utils.data.DataLoader(
#     testset2, batch_size=args.batch_size, shuffle=False, num_workers=2)
# test(net,testloader)
# test(net,testloader2)
# img,lab = next(iter(testloader2))
def test2(net,testloader):
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=args.lr,
    #                     momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate((testloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            # inputs2 = simba_single(model=net, x=inputs, y=targets, num_iters=10, epsilon=4./255)
            list = [0]*args.batch_size
            for index, (img, tar) in enumerate((zip(inputs, targets))):
                list[index] = simba_single(model=net, x=img.unsqueeze(0), y=tar, num_iters=3072, epsilon=8./255)
            inputs2 = torch.cat(list,dim=0).to(device)
            outputs = net(inputs2)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print( 'Loop: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (batch_idx+1, test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            if(batch_idx+1 == 250):
                break
        print( 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
test2(net,testloader)