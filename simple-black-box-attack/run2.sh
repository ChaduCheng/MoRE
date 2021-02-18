#!/bin/bash
trap ctrl_c INT
function ctrl_c(){
        echo "** Trapped CTRL-C"
        exit 1
}

cd /home/chenan/hao0205/simple-blackbox-attack
eval "$(conda shell.bash hook)"
conda activate msd
# CUDA_VISIBLE_DEVICES=3 python run_simba_l2.py -m 0 -v 0 -p
# CUDA_VISIBLE_DEVICES=3 python run_simba_l2.py -m 0 -v 1 -p

# CUDA_VISIBLE_DEVICES=3 python run_simba_linf.py -m 0 -v 0 -p
# CUDA_VISIBLE_DEVICES=3 python run_simba_linf.py -m 0 -v 1 -p

CUDA_VISIBLE_DEVICES=2 python run_simba_l2.py -m 1 -v 0 -p
CUDA_VISIBLE_DEVICES=2 python run_simba_l2.py -m 2 -v 1 -p

# CUDA_VISIBLE_DEVICES=3 python run_simba_linf.py -m 1 -v 0 -p
# CUDA_VISIBLE_DEVICES=3 python run_simba_linf.py -m 2 -v 1 -p
#############
# CUDA_VISIBLE_DEVICES=0 python run_simba_l2.py -m 0 -v 0 
# CUDA_VISIBLE_DEVICES=0 python run_simba_l2.py -m 0 -v 1 

# CUDA_VISIBLE_DEVICES=0 python run_simba_linf.py -m 0 -v 0 
# CUDA_VISIBLE_DEVICES=0 python run_simba_linf.py -m 0 -v 1 

# CUDA_VISIBLE_DEVICES=0 python run_simba_l2.py -m 1 -v 0 
# CUDA_VISIBLE_DEVICES=0 python run_simba_l2.py -m 2 -v 1 

# CUDA_VISIBLE_DEVICES=0 python run_simba_linf.py -m 1 -v 0 
# CUDA_VISIBLE_DEVICES=0 python run_simba_linf.py -m 2 -v 1 





