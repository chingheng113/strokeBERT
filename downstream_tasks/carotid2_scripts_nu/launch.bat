#!/bin/sh
swarm -f carotid2_sb_nu_0.swarm -g 100 -t 6 --job-name carotid2_sb_nu_0 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid2_sb_nu_1.swarm -g 100 -t 6 --job-name carotid2_sb_nu_1 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid2_sb_nu_2.swarm -g 100 -t 6 --job-name carotid2_sb_nu_2 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid2_sb_nu_3.swarm -g 100 -t 6 --job-name carotid2_sb_nu_3 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid2_sb_nu_4.swarm -g 100 -t 6 --job-name carotid2_sb_nu_4 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid2_sb_nu_5.swarm -g 100 -t 6 --job-name carotid2_sb_nu_5 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid2_sb_nu_6.swarm -g 100 -t 6 --job-name carotid2_sb_nu_6 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid2_sb_nu_7.swarm -g 100 -t 6 --job-name carotid2_sb_nu_7 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid2_sb_nu_8.swarm -g 100 -t 6 --job-name carotid2_sb_nu_8 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid2_sb_nu_9.swarm -g 100 -t 6 --job-name carotid2_sb_nu_9 --module python/3.6 --gres=gpu:p100:2 --partition=gpu