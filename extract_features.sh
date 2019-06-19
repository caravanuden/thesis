#!/bin/bash

# extract features from R(2+1)D model trained on Kinetics and Sports1M
# extracts the features for each layer and saves them separately
# run from /ihome/cara/cvu_thesis/thesis/VMZ
# CVU 2019

python data/create_video_db.py \
--list_file=/ihome/cara/cvu_thesis/thesis/raiders_clips.csv \
--output_file=/ihome/cara/cvu_thesis/raiders_db \
--use_list=1 --use_video_id=1 --use_start_frame=1; \

# --gpus= \
# --clip_length_rgb=32 \
python tools/extract_features.py \
--test_data=/ihome/cara/cvu_thesis/raiders_db \
--model_name=r2plus1d --model_depth=18 \
--clip_length_rgb=32 \
--sampling_rate_rgb=2 \
--batch_size=32 \
--load_model_path=/ihome/cara/cvu_thesis/cnn_models/r2.5d_d18_l32.pkl \
--output_path=/ihome/cara/cvu_thesis/activations/features_r2.5d_d18_l32.pkl \
--features=last_out_L400_w,conv1_middle_w,conv1_w,comp_1_conv_2_middle_w,comp_1_conv_2_w,comp_3_conv_2_middle_w,comp_3_conv_2_w,comp_5_conv_2_middle_w,comp_5_conv_2_w,comp_7_conv_2_middle_w,comp_7_conv_2_w \
--sanity_check=1 --get_video_id=1 --use_local_file=1 --num_labels=400; \
