import os
from argparse import ArgumentParser
#from testing import validation
#from trained_model_expert import train
from train_mod_adv import train
from train_mod_adv import test, test_deepfool


def get_args():
    parser = ArgumentParser(description='Mixture of Experts')
    # parser.add_argument('--epochs', type=int, default=100)
    # parser.add_argument('--dataset', type=float, default='mnist')
    # parser.add_argument('--batch_size', type=int, default=8)
    # parser.add_argument('--train_split', type=float, default=0.8)
    # parser.add_argument('--lr', type=float, default=.001)
    # parser.add_argument('--gpu_ids', type=str, default='0')
    # parser.add_argument('--checkpoint_loc', type=str, default='./checkpoint/latest_model.ckpt')
    # parser.add_argument('--num_experts', type=int, default=16)
    # parser.add_argument('--training', type=bool, default=True)
    # parser.add_argument('--testing', type=bool, default=False)
    args = parser.parse_args()
    args.epochs = 1
    args.dataset = 'cifar'
    args.batch_size = 1
    args.train_split = 0.05
    args.lr = 0.1
    args.gpu_ids = '0'
    # args.checkpoint_loc = '../cifar_16_resenet/checkpoint/ckptMoE_resnet_cifar_clean+4adv_final_ensure_noclean.pth'
    args.checkpoint_loc = '../cifar_16_resenet/checkpoint/ckptMoE_resnet_cifar_clean+4adv+4nat_final_ensure.pth'
    #args.checkpoint_loc = './checkpoint/ckptMoE_resnet_cifar_clean+4nat.pth'
    #args.checkpoint_loc = './checkpoint/ckptMoE_resnet_cifar_clean+4adv_adaptive.pth'
    #args.checkpoint_loc = './trained_model/ckptl2_alex_cifar_50.pth'
    args.num_experts = 9
    args.training = False
    args.testing = True
    args.norm = '2'
    args.epsilon = 1.0
    # args.tr       aining = False
    # args.testing = True
    return args

def main():
    args = get_args()
    
    if args.training:
        train(args)
    if args.testing:
        test_deepfool(args)


if __name__ == '__main__':
    main()