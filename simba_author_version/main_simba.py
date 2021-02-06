import os
from argparse import ArgumentParser
#from testing import validation
from test_simba import test
#from tests import test


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
    args.epochs = 100
    args.dataset = 'cifar'
    args.batch_size = 1000
    args.train_split = 0.8 
    args.lr = 0.1
    args.gpu_ids = '0'
    #args.checkpoint_loc = './checkpoint/ckptnat_resnet18_cifar_0.1_lu_nat_adv_fog.pth'
    #args.checkpoint_loc = './trained_model/ckptl2_alex_cifar_50.pth'
    args.num_experts = 9
    #args.training = True
    #args.testing = False
    # args.training = False
    # args.testing = True



    #args.sampled_image_dir = ''
    #args.model = 'resnet18'
    #args.num_runs = 1000
    #args.batch_size = 8
    # args.num_iters = 0
    # args.log_every = 10
    # args.epsilon = 0.314 * 5
    # args.linf_bound = 0.0
    # args.freq_dims = 14
    # args.order = ''  # only used in frequency attack
    # args.stride = 7
    # args.targeted = False
    # args.pixel_attack = True
    # args.save_suffix = ''
    # args.train_split = 0.8

    args.num_runs = 1000
    args.num_iters = 0
    args.log_every = 10
    args.epsilon = 0.2
    args.linf_bound = 0.0
    args.freq_dims =4
    args.order = ''  # only used in frequency attack
    args.stride = 7
    args.targeted = False
    args.pixel_attack = True
    #args.save_suffix = ''
    #args.train_split = 0.8
    return args

def main():
    args = get_args()
    test(args)
    # str_ids = args.gpu_ids.split(',')
    # args.gpu_ids = []
    # for str_id in str_ids:
    #     id = int(str_id)
    #     if id >= 0:
    #         args.gpu_ids.append(id)
    
    # if args.training:
    #
    # if args.testing:
    #     test(args)


if __name__ == '__main__':
    main()