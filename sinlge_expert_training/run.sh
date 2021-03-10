CUDA_VISIBLE_DEVICES=0 python main_desk_l2_lu.py -v 0 
CUDA_VISIBLE_DEVICES=1 python main_desk_l2_lu.py -v 1 
CUDA_VISIBLE_DEVICES=2 python main_desk_linf_lu.py -v 0 
CUDA_VISIBLE_DEVICES=3 python main_desk_linf_lu.py -v 1 

CUDA_VISIBLE_DEVICES=0 python -m robustness.main --dataset cifar --data ./data  --adv-train 1 --arch resnet18 --out-dir ./checkpoints --attack-steps 20 --constraint inf --eps 0.031372549 --random-restarts 1 --attack-lr 0.01 --adv-eval 1 --epochs 200
CUDA_VISIBLE_DEVICES=1 python -m robustness.main --dataset cifar --data ./data  --adv-train 1 --arch resnet18 --out-dir ./checkpoints --attack-steps 20 --constraint inf --eps 0.0235294118 --random-restarts 1 --attack-lr 0.01 --adv-eval 1 --epochs 200
CUDA_VISIBLE_DEVICES=2 python -m robustness.main --dataset cifar --data ./data  --adv-train 1 --arch resnet18 --out-dir ./checkpoints --attack-steps 20 --constraint 2 --eps 0.5 --random-restarts 1 --attack-lr 0.1 --adv-eval 1 --epochs 200
CUDA_VISIBLE_DEVICES=3 python -m robustness.main --dataset cifar --data ./data  --adv-train 1 --arch resnet18 --out-dir ./checkpoints --attack-steps 20 --constraint 2 --eps 1.0 --random-restarts 1 --attack-lr 0.2 --adv-eval 1 --epochs 200


CUDA_VISIBLE_DEVICES=0 python main_desk_l2_lu.py -v 0 --dataset tinyimagenet
CUDA_VISIBLE_DEVICES=1 python main_desk_l2_lu.py -v 1 --dataset tinyimagenet
CUDA_VISIBLE_DEVICES=2 python main_desk_linf_lu.py -v 0 --dataset tinyimagenet
CUDA_VISIBLE_DEVICES=3 python main_desk_linf_lu.py -v 1 --dataset tinyimagenet 
CUDA_VISIBLE_DEVICES=0 python main_desk_lu.py --dataset tinyimagenet 

CUDA_VISIBLE_DEVICES=0 python main_desk_l1_lu.py -v 0 
CUDA_VISIBLE_DEVICES=1 python main_desk_l1_lu.py -v 1 