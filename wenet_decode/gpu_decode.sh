#!/bin/bash

test_data=data/music/data.list
result_file=wenet_decode/exp/result.txt
wenet_dir=/Data/wenet
model_dir=/Data/models

checkpoint=$model_dir/20220506_u2pp_conformer_exp/final.pt 
dict=$model_dir/20220506_u2pp_conformer_exp/units.txt
config=$model_dir/20220506_u2pp_conformer_exp/train.yaml
mode=ctc_prefix_beam_search
ctc_weight=0.5

export PYTHONPATH=$wenet_dir
/home/v2129375/anaconda3/bin/python $wenet_dir/wenet/bin/recognize.py --gpu 0 \
            --mode $mode \
            --config $config \
            --data_type raw \
            --test_data $test_data \
            --checkpoint $checkpoint \
            --beam_size 10 \
            --batch_size 1 \
            --penalty 0.0 \
            --dict $dict \
            --ctc_weight $ctc_weight \
            --result_file $result_file \
            ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}