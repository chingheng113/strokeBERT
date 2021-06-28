#!/bin/sh
swarm -f carotid_sb_rr_0.swarm -g 100 -t 6 --job-name carotid_sb_rr_0 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid_sb_rr_1.swarm -g 100 -t 6 --job-name carotid_sb_rr_1 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid_sb_rr_2.swarm -g 100 -t 6 --job-name carotid_sb_rr_2 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid_sb_rr_3.swarm -g 100 -t 6 --job-name carotid_sb_rr_3 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid_sb_rr_4.swarm -g 100 -t 6 --job-name carotid_sb_rr_4 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid_sb_rr_5.swarm -g 100 -t 6 --job-name carotid_sb_rr_5 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid_sb_rr_6.swarm -g 100 -t 6 --job-name carotid_sb_rr_6 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid_sb_rr_7.swarm -g 100 -t 6 --job-name carotid_sb_rr_7 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid_sb_rr_8.swarm -g 100 -t 6 --job-name carotid_sb_rr_8 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid_sb_rr_9.swarm -g 100 -t 6 --job-name carotid_sb_rr_9 --module python/3.6 --gres=gpu:p100:2 --partition=gpu