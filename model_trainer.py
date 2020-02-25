import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import pandas as pd
import os
import json


class ModelTrainer:
    def __init__(self, parsed_args):
        self.data_dir = parsed_args.data_dir
        self.save_dir = parsed_args.save_dir
        self.arch = parsed_args.arch
        self.learning_rate = parsed_args.learning_rate
        self.hidden_units = parsed_args.hidden_units
        self.epochs = parsed_args.epochs
        self.use_gpu = parsed_args.gpu
        self._device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        print(f"Selected device type: {self._device}")

        self.train_dir = os.path.join(self.data_dir, 'train')
        self.valid_dir = os.path.join(self.data_dir + 'valid')
        self.test_dir = os.path.join(self.data_dir + 'test')

        data_transforms = self.build_data_transforms()
        self.image_datasets = {
            "train": datasets.ImageFolder(self.train_dir, transform=data_transforms["train"]),
            "test": datasets.ImageFolder(self.test_dir, transform=data_transforms["test"]),
            "validation": datasets.ImageFolder(self.valid_dir, transform=data_transforms["validation"])
        }
    
    def build_data_transforms(self):
        return {
                "train": transforms.Compose([
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "test": transforms.Compose([
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "validation": transforms.Compose([
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
    
    def select_model(self, turn_off_model_modifications=True):
        try:
            model = models.__dict__[self.arch](pretrained=True)
            if turn_off_model_modifications:
                for param in model.parameters():
                    param.requires_grad = False

            return model
        except KeyError:
            print(f"Architecture {self.arch} not found.")

    def build_model_classifier(self):
        return nn.Sequential(
            nn.Linear(25088, self.hidden_units),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.hidden_units, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 102),
            nn.LogSoftmax(dim=1)
        )

    def save_model_checkpoint(self, model):
        if self.save_dir:
            model.class_to_idx =  self.image_datasets['train'].class_to_idx
            model.to('cpu')
            checkpoint = {
                "name": self.arch,
                "classifier": model.classifier,
                "state_dict": model.state_dict(),
                "mapping": model.class_to_idx
            }
            checkpoint_path = os.path.join(self.save_dir, "train_checkpoint.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"Model checkpoint saved to: {checkpoint_path}")

    def train(self):
        trainloader, testloader, validationloader = self.build_data_loaders()
        model = self.select_model()
        classifier = self.build_model_classifier()

        model.classifier =  classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=self.learning_rate)

        self.train_and_check_accuracy(model, criterion, optimizer, trainloader, validationloader)

        self.save_model_checkpoint(model)
        
    
    def train_and_check_accuracy(self, model, criterion, optimizer, trainloader, validationloader):
        print("Starting model training...")
        steps = 0
        running_loss = 0
        print_every = 40
        
        
        model.to(self._device)

        for epoch in range(self.epochs):
            for images, labels in trainloader:
                steps += 1
                
                images, labels = images.to(self._device), labels.to(self._device)
                
                optimizer.zero_grad()
                
                logps = model.forward(images)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for images, labels in validationloader:
                            images, labels = images.to(self._device), labels.to(self._device)
                            logps = model.forward(images)
                            batch_loss = criterion(logps, labels)

                            test_loss += batch_loss.item()

                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                    print(f"Epoch {epoch+1}/{self.epochs}.. "
                        f"Train loss: {running_loss/print_every:.3f}.. "
                        f"Validation loss: {test_loss/len(validationloader):.3f}.. "
                        f"Validation accuracy: {accuracy/len(validationloader):.3f}")
                    running_loss = 0
                    model.train()

    def build_data_loaders(self):
        dataloaders = {
            "train": torch.utils.data.DataLoader(self.image_datasets["train"], batch_size=64, shuffle=True),
            "test": torch.utils.data.DataLoader(self.image_datasets["test"], batch_size=64),
            "validation": torch.utils.data.DataLoader(self.image_datasets["validation"], batch_size=64)
        }
        return dataloaders['train'], dataloaders['test'], dataloaders['validation']



