import os
from argparse import ArgumentParser
#from testing import validation
from train_mod_fog_ul import train
# from tests import test


def get_args():
    parser = ArgumentParser(description='Mixture of Experts')
    # parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='tinyimagenet')
    parser.add_argument('--heavy','-v', type=int, default=None)
    parser.add_argument('--out_dir','-o',type=str, default='./ckpt_resnet18/')
    # parser.add_argument('--batch_size', type=int, default=8)
    # parser.add_argument('--train_split', type=float, default=0.8)
    # parser.add_argument('--lr', type=float, default=.001)
    # parser.add_argument('--gpu_ids', type=str, default='0')
    # parser.add_argument('--checkpoint_loc', type=str, default='./checkpoint/latest_model.ckpt')
    # parser.add_argument('--num_experts', type=int, default=16)
    # parser.add_argument('--training', type=bool, default=True)
    # parser.add_argument('--testing', type=bool, default=False)
    args = parser.parse_args()
    if args.heavy is None:
        raise Exception("not implemented")
    args.epochs = 200
    # args.dataset = 'cifar'
    args.batch_size = 128
    args.train_split = 0.8 
    args.lr = 0.1
    args.gpu_ids = '0'
    # args.checkpoint_loc = './checkpoint/clean.pth'
    #args.checkpoint_loc = './trained_model/ckptl2_alex_cifar_50.pth'
    args.num_experts = 3
#     args.training = True
#     args.testing = False
    # args.training = False
    # args.testing = True
    return args

def main():
    args = get_args()
    # str_ids = args.gpu_ids.split(',')
    # args.gpu_ids = []
    # for str_id in str_ids:
    #     id = int(str_id)
    #     if id >= 0:
    #         args.gpu_ids.append(id)
    
#     if args.training:
    # print(args.checkpoint_loc)
    train(args)
    # test(args)
#     if args.testing:
#         test(args)


if __name__ == '__main__':
    main()
