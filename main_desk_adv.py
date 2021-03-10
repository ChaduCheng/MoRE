import os
from argparse import ArgumentParser
#from testing import validation
#from trained_model_expert import train
from train_mod_adv import train
from train_mod_adv import test


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
    args.epochs = 1    # epoch =1 when doing testing
    args.dataset = 'cifar'
    args.batch_size = 128
    args.train_split = 0.8 
    args.lr = 0.1
    args.gpu_ids = '0'
    args.checkpoint_loc = './checkpoint/ckptMoE_resnet_cifar_clean+4adv_true.pth'   # training phase:  save directory ;  testing pahse: testing model dir.
    #args.checkpoint_loc = './checkpoint/ckptMoE_resnet_cifar_clean+4nat.pth'
    #args.checkpoint_loc = './checkpoint/ckptMoE_resnet_cifar_clean+4adv_adaptive.pth'
    #args.checkpoint_loc = './trained_model/ckptl2_alex_cifar_50.pth'
    args.num_experts = 5 # useless when do expert trainin and testing
    args.training = True
    args.testing = False
    #args.norm = 'inf'   ## deep fool
    #args.epsilon = 8/255 ##deep fool 
    # args.norm = '2'   ## deepfool
    # args.epsilon = 1.0  ##deepfool
    args.usemodel = 'expert'   # expert: single expert;  more: MoRE systems
    args.seed = 1
    return args

def main():
    args = get_args()
    
    if args.training:
        train(args)
    if args.testing:
        test(args)


if __name__ == '__main__':
    main()
