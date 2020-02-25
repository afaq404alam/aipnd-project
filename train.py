from argparse import ArgumentParser
from torchvision import models

from model_trainer import ModelTrainer

arg_parser = ArgumentParser(description="Program for training a deep learning model")
arg_parser.add_argument("data_dir", help="Path to data directory to be used for training")
arg_parser.add_argument("--save_dir", help="Directory to save checkpoints")
arg_parser.add_argument("--arch", default="vgg16", help="Choose architecture")
arg_parser.add_argument("--learning_rate", type=float, default=0.001, help="Set learning rate")
arg_parser.add_argument("--hidden_units", type=int, default=4096, help="Set hidden units")
arg_parser.add_argument("--epochs", type=int, default=7, help="Set number of epochs")
arg_parser.add_argument("--gpu", action="store_true", default=False, help="Use GPU for training")

parsed_args = arg_parser.parse_args()

def train():
    model_trainer = ModelTrainer(parsed_args)
    model_trainer.train()


if __name__ == "__main__":
    train()

