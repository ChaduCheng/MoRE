CUDA_VISIBLE_DEVICES=1 python main_desk_fog_ul.py -v 0 --dataset tinyimagenet 
CUDA_VISIBLE_DEVICES=0 python main_desk_fog_ul.py -v 1 --dataset tinyimagenet 

CUDA_VISIBLE_DEVICES=0 python main_desk_snow_ul.py -v 0 --dataset tinyimagenet 
CUDA_VISIBLE_DEVICES=1 python main_desk_snow_ul.py -v 1 --dataset tinyimagenet 
CUDA_VISIBLE_DEVICES=2 python main_desk_rota_lu.py -v 0 --dataset tinyimagenet 
CUDA_VISIBLE_DEVICES=3 python main_desk_rota_lu.py -v 1 --dataset tinyimagenet 
