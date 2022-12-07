#!/bin/bash
opennmt_dir=/Data/OpenNMT-py

export PYTHONPATH=$opennmt_dir
/home/v2129375/anaconda3/bin/python $opennmt_dir/translate.py --model asp/exp/model.mt_zh/_step_100000.pt \
            --src data/video_test/keyword_jieba_space.txt \
            --output asp/exp/opennmt_out.txt \
            --gpu 0 \
            --verbose \
            --n_best 5 \
            --batch_size 1 \
            --with_score 


