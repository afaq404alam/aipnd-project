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

        #turning off tuning of the model
        for param in model.parameters(): 
            param.requires_grad = False
            
        model.classifier = checkpoint['classifier']
        model.class_to_idx = checkpoint['mapping']
        
        model.load_state_dict(checkpoint['state_dict'])
        
        return model

    def get_cat_to_name_map(self):
        with open(self.category_names, 'r') as f:
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
        preprocessed_img = preprocessed_img.unsqueeze_(0)
        preprocessed_img = preprocessed_img.float()
        preprocessed_img.to(self._device)



        with torch.no_grad():
            output = model(preprocessed_img)

        probs, classes = F.softmax(output.data, dim=1).topk(self.top_k)

        probs_classes = zip(probs.tolist()[0], classes.tolist()[0])

        cat_to_names = None
        if self.category_names:
            cat_to_names = self.get_cat_to_name_map()

        inv_map = {v: k for k, v in model.class_to_idx.items()}
        if cat_to_names:
            return [{"probability": prob, "class_id": inv_map[klass], "class_name": cat_to_names[inv_map[klass]]} for prob, klass in probs_classes]
        else:
            return [{"probability": prob, "class_id": inv_map[klass]} for prob, klass in probs_classes]
