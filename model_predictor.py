import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import pandas as pd
import os
import json
from PIL import Image

class ModelPredictor:
    def __init__(self, parsed_args):
        self.image_path = parsed_args.image_path
        self.checkpoint = parsed_args.checkpoint
        self.top_k = parsed_args.top_k
        self.category_names = parsed_args.category_names
        self.use_gpu = parsed_args.gpu
        self._device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        print(f"Selected device type: {self._device}")

    def load_model(self):
        print(f"Loading model from {self.checkpoint}...")
        checkpoint = torch.load(self.checkpoint)
        model_name = checkpoint['name']

        model = models.__dict__[model_name](pretrained = True)
            
        model.classifier = checkpoint['classifier']
        model.class_to_idx = checkpoint['mapping']
        
        model.load_state_dict(checkpoint['state_dict'])
        
        #turning off tuning of the model
        for param in model.parameters(): 
            param.requires_grad = False 
        
        return model

    def get_cat_to_name_map(self, cat_to_name_file_path: str):
        with open('cat_to_name.json', 'r') as f:
            return json.load(f)

    def preprocess_image(self):
        img = Image.open(self.image_path)

        transform = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return transform(img)

    def predict(self):
        model = self.load_model()
        model.to(self._device)

        preprocessed_img = self.preprocess_image()
        preprocessed_img.to(self._device)

        output = model(preprocessed_img)

        probs = F.softmax(output.data, dim=1)

        return probs.topk(self.top_k)
