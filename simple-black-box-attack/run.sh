CUDA_VISIBLE_DEVICES=1 python simba_single.py -p './checkpoint/ckptnat_resnet18_cifar_tsave.pth'
CUDA_VISIBLE_DEVICES=2 python simba_single.py -p './checkpoint/ckptnat_resnet18_cifar_0.1_lu_linf_6_255.pth'
CUDA_VISIBLE_DEVICES=1 python simba_single.py -p './checkpoint/ckptnat_resnet18_cifar_tsave.pth' --norm l2 --epsilon 60 --iteration 3072 -st 500
CUDA_VISIBLE_DEVICES=2 python simba_single.py -p './checkpoint/ckptnat_resnet18_cifar_tsave.pth' --norm linf --epsilon 6 --iteration 3072 -st 500



###############################
CUDA_VISIBLE_DEVICES=1 python simba_batch.py -p './checkpoint/ckptnat_resnet18_cifar_tsave.pth' --norm l2 --epsilon 60 --iteration -1 --batch_size 64
CUDA_VISIBLE_DEVICES=2 python simba_batch.py -p './checkpoint/ckptnat_resnet18_cifar_tsave.pth' --norm linf --epsilon 6 --iteration -1 --batch_size 64

CUDA_VISIBLE_DEVICES=0 python simba_batch.py -p "./checkpoint/ckptnat_resnet18_cifar_0.1_lu_l2_60_255.pth"  --norm l2 --epsilon 60 --iteration -1 --batch_size 64
CUDA_VISIBLE_DEVICES=3 python simba_batch.py -p './checkpoint/ckptnat_resnet18_cifar_0.1_lu_linf_6_255.pth' --norm linf --epsilon 6 --iteration -1 --batch_size 64
#######
CUDA_VISIBLE_DEVICES=2 python simba_batch.py -p './checkpoint/ckptnat_resnet18_cifar_tsave.pth' --norm linf --epsilon 6 --iteration -1 --batch_size 4
###############################
CUDA_VISIBLE_DEVICES=0 python simba_single_back.py -p './checkpoint/ckptnat_resnet18_cifar_tsave.pth' --norm l2 --epsilon 60 --batch_size 2 -m 2
CUDA_VISIBLE_DEVICES=1 python simba_single_back.py -p './checkpoint/ckptnat_resnet18_cifar_tsave.pth' --norm linf --epsilon 6 --batch_size 2 -m 2

CUDA_VISIBLE_DEVICES=2 python simba_single_back.py -p "./checkpoint/ckptnat_resnet18_cifar_0.1_lu_l2_60_255.pth"  --norm l2 --epsilon 60 --batch_size 2 -m 2
CUDA_VISIBLE_DEVICES=3 python simba_single_back.py -p './checkpoint/ckptnat_resnet18_cifar_0.1_lu_linf_6_255.pth' --norm linf --epsilon 6 --batch_size 2 -m 2
##########
CUDA_VISIBLE_DEVICES=2 python simba_single_back.py -p './checkpoint/ckptnat_resnet18_cifar_tsave.pth' --norm linf --epsilon 6 --iteration -1 --batch_size 2 -m 2
###########
CUDA_VISIBLE_DEVICES=0 python simba_single_back.py -p './checkpoint/ckptnat_resnet18_cifar_tsave.pth' --norm l2 --epsilon 60 --batch_size 2 -m 1
CUDA_VISIBLE_DEVICES=1 python simba_single_back.py -p './checkpoint/ckptnat_resnet18_cifar_tsave.pth' --norm linf --epsilon 6 --batch_size 2 -m 1

CUDA_VISIBLE_DEVICES=2 python simba_single_back.py -p "./checkpoint/ckptnat_resnet18_cifar_0.1_lu_l2_60_255.pth"  --norm l2 --epsilon 60 --batch_size 2 -m 1
CUDA_VISIBLE_DEVICES=3 python simba_single_back.py -p './checkpoint/ckptnat_resnet18_cifar_0.1_lu_linf_6_255.pth' --norm linf --epsilon 6 --batch_size 2 -m 1


