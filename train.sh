#---------------------------------------------------------------------------------------------------
# THUMOS14 Training
CUDA_VISIBLE_DEVICES=0 python main.py --max-seqlen 500 --lr 0.00005 --k 7 --dataset-name Thumos14reduced --path-dataset path/to/Thumos14 --num-class 20 --use-model STCL_THU  --max-iter 10000  --dataset SampleDataset --weight_decay 0.001 --model-name THU_STCL_best --seed 3552 --AWM BWA_fusion_dropout_feat_v2

#---------------------------------------------------------------------------------------------------
#ActivityNet Training
CUDA_VISIBLE_DEVICES=0 python main.py --k 5  --dataset-name ActivityNet1.2/1.3 --path-dataset path/to/ActivityNet dataset --num-class number-of-actions --use-model STCL_ACT  --dataset AntSampleDataset --lr 3e-5 --max-seqlen 60 --model-name ACT_STCL_best --seed 3552 --max-iter 40000