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
#SBATCH --exclude=viscam12,viscam1
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

source /atlas/u/sgu33/miniconda3/etc/profile.d/conda.sh
# Activate the llava conda environment
conda activate /atlas/u/sgu33/miniconda3/envs/llava

export HOME=/viscam/projects/GenLayout
#checkpoint_path=/viscam/projects/GenLayout/GenLayout_sun/third_party/LLaVa-1.6-ft/checkpoints/--finetune_task_lora
#fine_tuned_checkpoint_path = "/viscam/projects/GenLayout/GenLayout_sun/third_party/LLaVa-1.6-ft/checkpoints"
#version = str(sys.argv[2])

#if fine_tuned_checkpoint_path != "default":
#    fine_tuned_model_path = fine_tuned_checkpoint_path
#    model_name =  get_model_name_from_path(fine_tuned_model_path)
#    model_base = "liuhaotian/llava-v1.5-7b"
#    #model_base = "liuhaotian/llava-v1.5-13b"
#else:
#    #fine_tuned_model_path = "liuhaotian/llava-v1.5-7b"
#    #model_name =  "liuhaotian/llava-v1.5-7b"
#    #model_base = None
#    fine_tuned_model_path = "liuhaotian/llava-v1.6-mistral-7b"
#    model_name =  "liuhaotian/llava-1.6-mistral-7b"
#    model_base = None

data_file=/viscam/projects/GenLayout/GenLayout_sun/data/3dfront_test_sample.json
model_name="lmms-lab/llava-next-interleave-qwen-7b"
checkpoint_path="/viscam/u/sgu33/GenLayout/third_party/LLaVA-NeXT-Reproduced_backup/checkpoints/llava-next-interleave-7b-lora"
python run_eval.py $data_file $model_name $checkpoint_path
