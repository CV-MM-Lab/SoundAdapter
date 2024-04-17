
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from transformers import ClapModel, ClapProcessor
from models.model_vit_320_4_sd import ViT

def get_audio_embed(audio,adapter,model,processor):
    inputs = processor(audios=audio, return_tensors="pt",sampling_rate=48000).to("cuda")
    audio_embed_input = model.get_audio_features(**inputs)
    audio_embed_input = audio_embed_input.unsqueeze(0).to("cuda")
    embed = adapter(audio_embed_input)
    return embed

def init_adapter():
    path_pt = "../pretrained_models/SD_checkpoint.pth.tar"
    checkpoint = torch.load(path_pt,map_location='cuda')
    audio_adapter = ViT(512, 512,300, 8, 64)
    audio_adapter.load_state_dict(checkpoint['state_dict'])
    audio_adapter = audio_adapter.to("cuda")
    audio_adapter.eval()
    for p in audio_adapter.parameters():
        p.requires_grad = False
    return audio_adapter
def init_clap():
    model = ClapModel.from_pretrained("../pretrained_models/clap-htsat-fused").to("cuda")
    processor = ClapProcessor.from_pretrained("../pretrained_models/clap-htsat-fused")
    return model,processor

class SoundStableDiffusionPipline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        audio_embed: Optional[torch.FloatTensor] = None
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        c = audio_embed
        c_txt=prompt_embeds
        # for i_signal in range(3, 8):
        i_signal = 4
        c_font = c[:, 0:1, :]
        c_salient = torch.cat((c[:, 1:i_signal, :], c_txt[:, 1:8, :]), dim=1)
        # c_salient = torch.cat((c_txt[:, :i_signal, :], c[:, :i_signal, :]), dim=1)
        c_back = (c[:, 11:, :] + c_txt[:, 11:, :]) / 2
        c_in = torch.cat((c_font,c_salient, c_back), dim=1)
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, c_in])
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)


                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents
        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        return image[0]