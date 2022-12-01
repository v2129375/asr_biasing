import sys
import wenetruntime as wenet

wav_file = "/Data/dataset/catslu_traindev/data/video/audios/5151dfca6e935c111ab3fc9826df4ce0_59bbb8833327930da300002c.wav"
decoder = wenet.Decoder(lang='chs',
                        model_dir="/Data/models/20220506_u2pp_conformer_libtorch")
ans = decoder.decode_wav(wav_file)
print(ans)