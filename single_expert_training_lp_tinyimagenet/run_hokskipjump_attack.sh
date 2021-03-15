CUDA_VISIBLE_DEVICES=3 python main_desk_jump.py --dataset tinyimagenet --checkpoint_loc ./ckpt_resnet18/clean.pth
CUDA_VISIBLE_DEVICES=3 python main_desk_jump.py --dataset tinyimagenet --checkpoint_loc ./ckpt_resnet18/l2_0.5.pt
CUDA_VISIBLE_DEVICES=3 python main_desk_jump.py --dataset tinyimagenet --checkpoint_loc ./ckpt_resnet18/l2_1.0.pt
CUDA_VISIBLE_DEVICES=3 python main_desk_jump.py --dataset tinyimagenet --checkpoint_loc ./ckpt_resnet18/linf_6.pt
CUDA_VISIBLE_DEVICES=3 python main_desk_jump.py --dataset tinyimagenet --checkpoint_loc ./ckpt_resnet18/linf_8.pt
CUDA_VISIBLE_DEVICES=3 python main_desk_jump.py --dataset tinyimagenet --checkpoint_loc ./ckpt_resnet18/tinyimagenet_l1_8.0.pt
CUDA_VISIBLE_DEVICES=3 python main_desk_jump.py --dataset tinyimagenet --checkpoint_loc ./ckpt_resnet18/tinyimagenet_l1_12.0.pt

