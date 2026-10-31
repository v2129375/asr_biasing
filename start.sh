# 实验一：Kw3000Keywords，提示里放全部 4923 个地图关键词（约 2.5 小时）
python asr/phi4_map_zeroshot.py \
    --model_path asr/model/Kw3000Keywords \
    --conditions full

# 实验二：KworigKeywords，不给关键词（约 20 分钟）
python asr/phi4_map_zeroshot.py \
    --model_path asr/model/KworigKeywords \
    --conditions none