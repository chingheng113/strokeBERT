#!/bin/sh
swarm -f carotid_sb_all_0.swarm -g 100 -t 6 --job-name carotid_sb_all_0 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid_sb_all_1.swarm -g 100 -t 6 --job-name carotid_sb_all_1 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid_sb_all_2.swarm -g 100 -t 6 --job-name carotid_sb_all_2 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid_sb_all_3.swarm -g 100 -t 6 --job-name carotid_sb_all_3 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid_sb_all_4.swarm -g 100 -t 6 --job-name carotid_sb_all_4 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid_sb_all_5.swarm -g 100 -t 6 --job-name carotid_sb_all_5 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid_sb_all_6.swarm -g 100 -t 6 --job-name carotid_sb_all_6 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid_sb_all_7.swarm -g 100 -t 6 --job-name carotid_sb_all_7 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid_sb_all_8.swarm -g 100 -t 6 --job-name carotid_sb_all_8 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid_sb_all_9.swarm -g 100 -t 6 --job-name carotid_sb_all_9 --module python/3.6 --gres=gpu:p100:2 --partition=gpu