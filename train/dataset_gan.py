import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

import torchvision.transforms.functional as F
import clip

from transformers import CLIPTextModel, CLIPTokenizer

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):

        random.shuffle(root)
        self.pt = root
        
        self.nSamples = len(self.pt)
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_sttize = batch_size
        self.num_workers = num_workers
        '''
        
        # 加载text encoder
        self.text_encoder = CLIPTextModel.from_pretrained("../../sd/stable-diffusion-v1-5", subfolder="text_encoder").to("cuda")
        self.tokenizer = CLIPTokenizer.from_pretrained("../../sd/stable-diffusion-v1-5", subfolder="tokenizer")
        '''
        # CILPmodel
        self.clipmodel, self.clippreprocess = clip.load("ViT-B/32", device="cuda")


        
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 

        with torch.no_grad():
            text_gt = self.pt[index][0]
            audio_embed = self.pt[index][1]
            '''
            text_input = self.tokenizer(text_gt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True,
                                   return_tensors="pt")
            text_embeddings = self.text_encoder(text_input.input_ids.to("cuda"))[0]  # torch.Size([1, 77, 768])
            '''

            text_token = clip.tokenize(text_gt)
            text_embeddings = self.clipmodel.encode_text(text_token.to("cuda")).float()

        return audio_embed,text_embeddings