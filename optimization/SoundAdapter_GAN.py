import argparse
import math
import os
from librosa.core import audio

import torch
import torchvision
from torch import optim
from tqdm import tqdm
import sys
import librosa
import numpy as np
import random
import torch.nn.functional as F

import cv2

sys.path.append("./")
from criteria.soundclip_loss import SoundCLIPLoss,SoundCLIPCLAPLoss
from criteria.id_loss import IDLoss
from models.stylegan2.model import Generator
from utils import ensure_checkpoint_exists
from models.model_vit_320_4_gan import ViT
from transformers import pipeline,ClapModel, ClapProcessor


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def main(args):
    ensure_checkpoint_exists(args.ckpt)
    y, sr = librosa.load(args.audio_path, sr=44100)
    path = args.soundadapter
    checkpoint = torch.load(path, map_location='cuda')
    audio_adapter = ViT(512, 512, 100, 8, 64)
    audio_adapter.load_state_dict(checkpoint['state_dict'])
    audio_adapter = audio_adapter.to("cuda")
    audio_adapter.eval()
    for p in audio_adapter.parameters():
        p.requires_grad = False
    clapmodel = ClapModel.from_pretrained("laion/clap-htsat-fused").to("cuda")
    clapprocessor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
    inputs = clapprocessor(audios=y, return_tensors="pt",sampling_rate=48000).to("cuda")
    audio_embed_input = clapmodel.get_audio_features(**inputs)
    audio_embed_clap = audio_embed_input.unsqueeze(0).to("cuda")
    embed_clap = audio_adapter(audio_embed_clap)
    os.makedirs(args.results_dir, exist_ok=True)
    g_ema = Generator(args.stylegan_size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()

    mean_latent = g_ema.mean_latent(4096)
    if args.latent_path:
        latent_code_init = torch.load(args.latent_path).cuda()
        with torch.no_grad():
            img_orig, _ = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False)

    elif args.mode == "edit":
        latent_code_init_not_trunc = torch.randn(1, 512).cuda()
        with torch.no_grad():
            img_orig, latent_code_init = g_ema([latent_code_init_not_trunc], return_latents=True,
                                               truncation=args.truncation, truncation_latent=mean_latent)

    else:
        latent_code_init = mean_latent.detach().clone().repeat(1, 18, 1)

    latent = latent_code_init.detach().clone()
    latent.requires_grad = True
    soundclap_loss = SoundCLIPCLAPLoss(args)
    id_loss = IDLoss(args)
    optimizer = optim.Adam([latent], lr=args.lr)
    pbar = tqdm(range(args.step))

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr

        img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False)

        cosine_distance_loss = soundclap_loss(img_gen, embed_clap)

        similarity_loss = ((latent_code_init - latent) ** 2).sum()

        loss = args.lambda_similarity * similarity_loss + cosine_distance_loss + args.lambda_identity * id_loss(img_orig, img_gen)[0]

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        pbar.set_description(
            (
                f"loss: {loss.item():.4f};"
            )
        )
        if args.save_intermediate_image_every > 0 and i % args.save_intermediate_image_every == 0:
            with torch.no_grad():
                img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False)
            torchvision.utils.save_image(img_gen, f"results-1/{str(i).zfill(5)}.png", normalize=True, range=(-1, 1))

    if args.mode == "edit":
        with torch.no_grad():
            img_orig, _ = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False)

        final_result = torch.cat([img_orig, img_gen])
    else:
        final_result = img_gen

    return final_result


if __name__ == "__main__":
    seed = 0
    torch.manual_seed(seed)
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", type=str, default="sobbing person",
                        help="the text that guides the editing/generation")
    parser.add_argument("--audio_path", type=str, default="audio/laughing.wav")
    parser.add_argument("--soundadapter", type=str, default="pretrained_models/GAN_checkpoint.pth.tar")
    parser.add_argument("--ckpt", type=str, default="pretrained_models/stylegan2-ffhq-config-f.pt",
                        help="pretrained StyleGAN2 weights")
    parser.add_argument("--stylegan_size", type=int, default=1024, help="StyleGAN resolution")
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--step", type=int, default=300, help="number of optimization steps")
    parser.add_argument("--mode", type=str, default="edit", choices=["edit", "free_generation"],
                        help="choose between edit an image an generate a free one")
    parser.add_argument("--lambda_similarity", type=float, default=0.002,
                        help="weight of the latent distance (used for editing only)")
    parser.add_argument("--lambda_identity", type=float, default=0.00, help="weight of the identity loss")
    parser.add_argument("--latent_path", type=str, default=None,
                        help="starts the optimization from the given latent code if provided. Otherwose, starts from"
                             "the mean latent in a free generation, and from a random one in editing. "
                             "Expects a .pt format")
    parser.add_argument("--truncation", type=float, default=0.7,
                        help="used only for the initial latent vector, and only when a latent code path is"
                             "not provided")
    parser.add_argument("--save_intermediate_image_every", type=int, default=20,
                        help="if > 0 then saves intermidate results during the optimization")
    parser.add_argument("--results_dir", type=str, default="results-1")
    parser.add_argument("--adaptive_layer_masking", type=bool, default=False)
    parser.add_argument("--save_latent_path", type=str, default=None)
    parser.add_argument("--save_source_image_path", type=str, default=None)
    parser.add_argument("--save_manipulated_image_path", type=str, default=None)
    parser.add_argument("--save_manipulated_latent_code_path", type=str, default=None)

    args = parser.parse_args()

    result_image = main(args)

    torchvision.utils.save_image(result_image.detach().cpu(), os.path.join(args.results_dir, "final_result.jpg"),
                                 normalize=True, scale_each=True, range=(-1, 1))


