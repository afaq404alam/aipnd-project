from argparse import ArgumentParser
from torchvision import models

from model_predictor import ModelPredictor

arg_parser = ArgumentParser(description="Program for predict image category using a deep learning model")
arg_parser.add_argument("image_path", help="Path to image for prediction")
arg_parser.add_argument("checkpoint", help="Checkpoint path from where to load the model")
arg_parser.add_argument("--top_k", type=int, default=1, help="Return top K most likely classes")
arg_parser.add_argument("--category_names", help="Use a mapping of categories to real names")
arg_parser.add_argument("--gpu", action="store_true", default=False, help="Use GPU for training")

parsed_args = arg_parser.parse_args()

def predict():
    model_predictor = ModelPredictor(parsed_args)
    prediction = model_predictor.predict()

    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    predict()
