#!/bin/bash
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling
#################
#partition name
#SBATCH --partition=viscam
#################
#number of GPUs
#SBATCH --gres=gpu:l40s:6
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --account=viscam
#################
#set a job name
#SBATCH --job-name="original_v1.6_llava"
#################
#a file for job output, you can check job progress, append the job ID with %j to make it unique
#SBATCH --output=/viscam/projects/GenLayout/slurm_sbatch_sweep_out/%x.%j.out
#################
# a file for errors from the job
#SBATCH --error=/viscam/projects/GenLayout/slurm_sbatch_sweep_out/%x.%j.err

#################
#time you think you need; default is 2 hours
#format could be dd-hh:mm:ss, hh:mm:ss, mm:ss, or mm, 144
#SBATCH --time=13-23:59:00
#################
# Quality of Service (QOS); think of it as sending your job into a special queue; --qos=long for with a max job length of 7 days.
# uncomment ##SBATCH --qos=long if you want your job to run longer than 48 hours, which is the default for normal partition,
# NOTE- in the hns partition the default max run time is 7 days , so you wont need to include qos, also change to normal partition
# since dev max run time is 2 hours.
##SBATCH --qos=long
# We are submitting to the dev partition, there are several on sherlock: normal, gpu, bigmem (jobs requiring >64Gigs RAM)
##SBATCH -p dev
#################
# --mem is memory per node; default is 4000 MB per CPU, remember to ask for enough mem to match your CPU request, since
# sherlock automatically allocates 4 Gigs of RAM/CPU, if you ask for 8 CPUs you will get 32 Gigs of RAM, so either
# leave --mem commented out or request >= to the RAM needed for your CPU request.  It will also accept mem. in units, ie "--mem=4G"
#SBATCH --mem=32G
#################
# Have SLURM send you an email when the job ends or fails, careful, the email could end up in your clutter folder
# Also, if you submit hundreds of jobs at once you will get hundreds of emails.
# list out some useful information

export HOME=/viscam/u/sgu33/GenLayout
source /atlas/u/sgu33/miniconda3/etc/profile.d/conda.sh
conda activate llava
echo "activated"

RUN_NAME='llava-next-interleave-7b-sft'
version=3dfront_for_vlm_all_v0_short_cl_v3

working_directory=/viscam/u/sgu33/GenLayout/third_party/LLaVA-NeXT-Reproduced
data_path=/viscam/projects/GenLayout/GenLayout_sun/data/$version.json

which python

nnodes=1
num_gpus=6


cd $working_directory

export PYTHONPATH=/viscam/u/sgu33/GenLayout/third_party/LLaVA-NeXT-Reproduced:$PYTHONPATH

export WANDB_API_KEY='1b0ca0f0dc2e90d460addf9ccaeda83188ad001d'

deepspeed --num_nodes ${nnodes} --num_gpus ${num_gpus} --master_port=12700 llava/train/train_mem.py \
    --lora_enable True --lora_r 16 --lora_alpha 32 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmms-lab/llava-next-interleave-qwen-7b \
    --version qwen_1_5 \
    --data_path ${data_path} \
    --image_folder / \
    --unfreeze_mm_vision_tower False \
    --mm_vision_tower_lr 2e-6 \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio pad \
    --output_dir ./checkpoints/${RUN_NAME} \
    --mm_patch_merge_type flat \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ${RUN_NAME}
