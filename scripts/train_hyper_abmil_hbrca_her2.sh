#!/bin/bash
task="BRCA_HER2"
data_root_dir="./data/TCGA-BRCA/patch_512/conch_512_4096_con"
split_suffix='sitepre5_fold3'
exp_code="brca_her2_sitepre5_fold3"
model_type="hypermil"
base_mil="abmil"
test_name="hyper_abmil_conch_512_4096"
len_learnable_prompt=8
slide_align=1
pretrain_epoch=0
max_epochs=20

CUDA_VISIBLE_DEVICES=0  \
python train.py  \
--drop_out  \
--lr 2e-4  \
--reg 1e-4 \
--num_workers 4 \
--k 15  \
--label_frac 1    \
--label_num 0    \
--weighted_sample   \
--bag_loss ce   \
--inst_loss svm \
--task $task   \
--len_learnable_prompt $len_learnable_prompt  \
--base_mil $base_mil  \
--slide_align $slide_align  \
--pretrain_epoch $pretrain_epoch  \
--max_epochs $max_epochs  \
--log_data  \
--data_root_dir $data_root_dir   \
--exp_code $exp_code   \
--model_type $model_type    \
--test_name $test_name  \
--split_suffix $split_suffix  \
