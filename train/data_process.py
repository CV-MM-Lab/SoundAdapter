import torch
import librosa

import os
import argparse
from transformers import ClapModel, ClapProcessor


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--vgg_path", type=str, default="./vggsound/")
    args = parser.parse_args()

    model = ClapModel.from_pretrained("laion/clap-htsat-fused").to("cuda")
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")



    path = args.vgg_path
    pathdirs = os.listdir(path)
    audio_embedings = []

    for idx, pathdir in enumerate(pathdirs):
        list_sampe = []
        lab = pathdir.split("_")[0]
        y, sr = librosa.load(path+pathdir, sr=44100)
        inputs = processor(audios=y, return_tensors="pt", sampling_rate=48000).to("cuda")
        audio_embed_input = model.get_audio_features(**inputs)
        audio_embed_input = audio_embed_input
        audio_embed = audio_embed_input.detach()
        list_sampe.append(lab)
        list_sampe.append(audio_embed)
        audio_embedings.append(list_sampe)
    torch.save(audio_embedings, "vgg_new_total_3.pth")
    print(len(audio_embedings))



