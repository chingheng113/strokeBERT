#!/bin/sh
swarm -f carotid2_sb_ds_0.swarm -g 100 -t 6 --job-name carotid2_sb_ds_0 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid2_sb_ds_1.swarm -g 100 -t 6 --job-name carotid2_sb_ds_1 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid2_sb_ds_2.swarm -g 100 -t 6 --job-name carotid2_sb_ds_2 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid2_sb_ds_3.swarm -g 100 -t 6 --job-name carotid2_sb_ds_3 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid2_sb_ds_4.swarm -g 100 -t 6 --job-name carotid2_sb_ds_4 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid2_sb_ds_5.swarm -g 100 -t 6 --job-name carotid2_sb_ds_5 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid2_sb_ds_6.swarm -g 100 -t 6 --job-name carotid2_sb_ds_6 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid2_sb_ds_7.swarm -g 100 -t 6 --job-name carotid2_sb_ds_7 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid2_sb_ds_8.swarm -g 100 -t 6 --job-name carotid2_sb_ds_8 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid2_sb_ds_9.swarm -g 100 -t 6 --job-name carotid2_sb_ds_9 --module python/3.6 --gres=gpu:p100:2 --partition=gpu