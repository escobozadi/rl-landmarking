#!/bin/bash
source /home/diana/anaconda3/etc/profile.d/conda.sh
conda activate rl-env
python DQN.py --task train --files data/filenames/images.txt data/filenames/landmarks.txt --val_files data/filenames/images_val.txt data/filenames/landmarks_val.txt --model_name CommNet --file_type joint --landmarks 1 1 1 --memory_size 5000 --init_memory_size 1000 --max_episodes 5000 --save_freq 200 --steps_per_episode 100 --lr 0.001 --scheduler_step_size 10 --viz 0 --write --multiscale --attention --log_comment changerl --target_update_freq 5
