python asr/phi4_general_asr.py --model_path microsoft/Phi-4-multimodal-instruct
python asr/phi4_general_asr.py --model_path asr/model/KworigKeywords
python asr/compare_forgetting.py \
    --base asr/exp/forgetting/Phi-4-multimodal-instruct \
    --ft asr/exp/forgetting/KworigKeywords