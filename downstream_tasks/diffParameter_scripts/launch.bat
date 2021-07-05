#!/bin/sh
swarm -f carotid_24.swarm -g 100 -t 6 --job-name carotid_24 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid_25.swarm -g 100 -t 6 --job-name carotid_25 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid_26.swarm -g 100 -t 6 --job-name carotid_26 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid_34.swarm -g 100 -t 6 --job-name carotid_34 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid_35.swarm -g 100 -t 6 --job-name carotid_35 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid_36.swarm -g 100 -t 6 --job-name carotid_36 --module python/3.6 --gres=gpu:p100:2 --partition=gpu

swarm -f carotid2_24.swarm -g 100 -t 6 --job-name carotid2_24 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid2_25.swarm -g 100 -t 6 --job-name carotid2_25 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid2_26.swarm -g 100 -t 6 --job-name carotid2_26 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid2_34.swarm -g 100 -t 6 --job-name carotid2_34 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid2_35.swarm -g 100 -t 6 --job-name carotid2_35 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f carotid2_36.swarm -g 100 -t 6 --job-name carotid2_36 --module python/3.6 --gres=gpu:p100:2 --partition=gpu

swarm -f restroke_24.swarm -g 100 -t 6 --job-name restroke_24 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f restroke_25.swarm -g 100 -t 6 --job-name restroke_25 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f restroke_26.swarm -g 100 -t 6 --job-name restroke_26 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f restroke_34.swarm -g 100 -t 6 --job-name restroke_34 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f restroke_35.swarm -g 100 -t 6 --job-name restroke_35 --module python/3.6 --gres=gpu:p100:2 --partition=gpu
swarm -f restroke_36.swarm -g 100 -t 6 --job-name restroke_36 --module python/3.6 --gres=gpu:p100:2 --partition=gpu