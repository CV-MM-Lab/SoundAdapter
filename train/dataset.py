import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset

from transformers import CLIPTextModel, CLIPTokenizer

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        #if train:
        #    root = root *4
        # self.text_gt = root['labs']
        # self.audio_embedings = root["audio_embedings"]
        random.shuffle(root)
        self.pt = root
        
        self.nSamples = len(self.pt)
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_sttize = batch_size
        self.num_workers = num_workers

        # 加载text encoder
        self.text_encoder = CLIPTextModel.from_pretrained("../../sd/stable-diffusion-v1-5", subfolder="text_encoder").to("cuda")
        self.tokenizer = CLIPTokenizer.from_pretrained("../../sd/stable-diffusion-v1-5", subfolder="tokenizer")

        
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        
        #img_path = self.lines[index]
        
        #img,target = load_data(img_path,self.train)
        
        #img = 255.0 * F.to_tensor(img)
        
        #img[0,:,:]=img[0,:,:]-92.8207477031
        #img[1,:,:]=img[1,:,:]-95.2757037428
        #img[2,:,:]=img[2,:,:]-104.877445883


        
        
        #if self.transform is not None:
        #    img = self.transform(img)
        with torch.no_grad():
            text_gt = self.pt[index][0]
            audio_embed = self.pt[index][1]

            text_input = self.tokenizer(text_gt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True,
                                   return_tensors="pt")
            text_embeddings = self.text_encoder(text_input.input_ids.to("cuda"))[0]  # torch.Size([1, 77, 768])
            # text_token = clip.tokenize(text_gt)
            # text_embeddings = self.clipmodel.encode_text(text_token.to("cuda")).float()
        #audio_embed = audio_embed / audio_embed.norm(dim=-1, keepdim=True)
            #text_embeddings = text_embeddings/text_embeddings.norm(dim=-1, keepdim=True)

        return audio_embed,text_embeddings