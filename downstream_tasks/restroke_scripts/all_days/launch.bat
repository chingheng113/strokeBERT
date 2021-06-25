#!/bin/sh
swarm -f restroke_sb_all_0.swarm -g 100 -t 6 --job-name restroke_sb_all_0 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f restroke_sb_all_1.swarm -g 100 -t 6 --job-name restroke_sb_all_1 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f restroke_sb_all_2.swarm -g 100 -t 6 --job-name restroke_sb_all_2 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f restroke_sb_all_3.swarm -g 100 -t 6 --job-name restroke_sb_all_3 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f restroke_sb_all_4.swarm -g 100 -t 6 --job-name restroke_sb_all_4 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f restroke_sb_all_5.swarm -g 100 -t 6 --job-name restroke_sb_all_5 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f restroke_sb_all_6.swarm -g 100 -t 6 --job-name restroke_sb_all_6 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f restroke_sb_all_7.swarm -g 100 -t 6 --job-name restroke_sb_all_7 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f restroke_sb_all_8.swarm -g 100 -t 6 --job-name restroke_sb_all_8 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f restroke_sb_all_9.swarm -g 100 -t 6 --job-name restroke_sb_all_9 --module python/3.6 --gres=gpu:p100:2 --partition=gpu

#swarm -f restroke_b_all_0.swarm -g 100 -t 6 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
#swarm -f restroke_b_all_1.swarm -g 100 -t 6 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
#swarm -f restroke_b_all_2.swarm -g 100 -t 6 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
#swarm -f restroke_b_all_3.swarm -g 100 -t 6 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
#swarm -f restroke_b_all_4.swarm -g 100 -t 6 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
#swarm -f restroke_b_all_5.swarm -g 100 -t 6 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
#swarm -f restroke_b_all_6.swarm -g 100 -t 6 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
#swarm -f restroke_b_all_7.swarm -g 100 -t 6 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
#swarm -f restroke_b_all_8.swarm -g 100 -t 6 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
#swarm -f restroke_b_all_9.swarm -g 100 -t 6 --module python/3.6 --gres=gpu:p100:2 --partition=gpu

#swarm -f restroke_c_all_0.swarm -g 100 -t 6 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
#swarm -f restroke_c_all_1.swarm -g 100 -t 6 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
#swarm -f restroke_c_all_2.swarm -g 100 -t 6 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
#swarm -f restroke_c_all_3.swarm -g 100 -t 6 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
#swarm -f restroke_c_all_4.swarm -g 100 -t 6 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
#swarm -f restroke_c_all_5.swarm -g 100 -t 6 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
#swarm -f restroke_c_all_6.swarm -g 100 -t 6 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
#swarm -f restroke_c_all_7.swarm -g 100 -t 6 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
#swarm -f restroke_c_all_8.swarm -g 100 -t 6 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
#swarm -f restroke_c_all_9.swarm -g 100 -t 6 --module python/3.6 --gres=gpu:p100:2 --partition=gpu