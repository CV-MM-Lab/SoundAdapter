from models.SoundStableDiffusion import SoundStableDiffusionPipline,init_clap,get_audio_embed,init_adapter
import torch
import librosa
import argparse



if __name__ == '__main__':
    seed = 0
    torch.manual_seed(seed)
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, default="audio/cat purring_3497.wav")
    parser.add_argument("--soundadapter",type=str, default="pretrained_models/SD_checkpoint.pth.tar")

    args = parser.parse_args()
    pipe = SoundStableDiffusionPipline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe = pipe.to("cuda")
    adapter = init_adapter(args.soundadapter)
    model, processor = init_clap()
    audio_1, _ = librosa.load(args.audio_path, sr=44100)
    embed_1 = get_audio_embed(audio_1, adapter, model, processor)
    image = pipe(audio_embed=embed_1)
    image.save('cat.png')