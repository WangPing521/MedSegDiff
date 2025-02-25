#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=def-chdesa
declare -a StringArray=(

#"python segmentation_train.py --data_name MMWHS --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma False --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 200 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 2"

"python segmentation_sample.py --data_name MMWHS --image_size 256 --model_path results/emasavedmodel_0.9999_015000.pt --out_dir ema15 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma False --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 200 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --batch_size 1"
"python segmentation_sample.py --data_name MMWHS --image_size 256 --model_path results/emasavedmodel_0.9999_020000.pt --out_dir ema20 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma False --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 200 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --batch_size 1"

"python segmentation_sample.py --data_name MMWHS --image_size 256 --model_path results/savedmodel015000.pt --out_dir model15 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma False --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 200 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --batch_size 1"
"python segmentation_sample.py --data_name MMWHS --image_size 256 --model_path results/savedmodel020000.pt --out_dir model20 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma False --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 200 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --batch_size 1"

)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
