from models.SoundStableDiffusion import SoundStableDiffusionPipline,init_clap,get_audio_embed,init_adapter
import torch
import librosa
import argparse



if __name__ == '__main__':
    seed = 0
    torch.manual_seed(seed)
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path_1", type=str, default="audio/helicopter_979.wav")
    parser.add_argument("--audio_path_2", type=str, default="audio/sea.wav")
    parser.add_argument("--soundadapter", type=str, default="pretrained_models/SD_checkpoint.pth.tar")

    args = parser.parse_args()

    pipe = SoundStableDiffusionPipline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe = pipe.to("cuda")
    adapter = init_adapter()
    model, processor = init_clap()
    audio_1, _ = librosa.load(args.audio_path_1, sr=44100)
    embed_1 = get_audio_embed(audio_1, adapter, model, processor)
    audio_2, _ = librosa.load(args.audio_path_2, sr=44100)  #
    embed_2 = get_audio_embed(audio_2, adapter, model, processor)
    c = embed_1
    c_txt = embed_2
    # for i_signal in range(3, 8):
    i_signal = 4
    c_font = c[:, 0:1, :]
    c_salient = torch.cat((c[:, 1:i_signal, :], c_txt[:, 1:3, :]), dim=1)
    # c_salient = torch.cat((c_txt[:, :i_signal, :], c[:, :i_signal, :]), dim=1)
    c_back = (c[:, 5:, :] + c_txt[:, 5:, :]) / 2
    c_in = torch.cat((c_font, c_salient, c_back), dim=1)
    image = pipe(audio_embed=c_in)
    image.save('mix.png')