#!/bin/bash

# extract features from R(2+1)D model trained on Kinetics and Sports1M
# extracts the features for each layer and saves them separately
# CVU 2019

python VMZ/data/create_video_db.py \
--list_file=raiders_clips.csv \
--output_file=raiders_db \
--use_list=1 --use_video_id=1 --use_start_frame=1; \

python VMZ/tools/extract_features.py \
--test_data=raiders_db \
--model_name=r2plus1d --model_depth=34 --clip_length_rgb=32 \
--gpus=0,1,2,3,4,5,6,7 \
--batch_size=4 \
--load_model_path=cnn_models/r2.5d_d34_l32_ft_sports1m.pkl \
--output_path=activations/r2.5d_d34_l32_ft_sports1m.pkl \
--features=softmax,label,video_id \
--sanity_check=1 --get_video_id=1 --use_local_file=1 --num_labels=400; \
