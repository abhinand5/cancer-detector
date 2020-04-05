import numpy as np
import cv2
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
from model import CancerDetector
from pathlib import Path

# Define Test Tranformations
test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Function to Predict Cancer
def predict(image):
    # Initialize Directories
    model_dir = Path("./saved_models/")
    # Load the CNN Model
    model = CancerDetector()
    # Load Model Weights
    model.load_state_dict(torch.load(model_dir/'best_model.pt', map_location=torch.device('cpu')))
    # Make model ready for inference
    model.eval()
    # Convert image to Tensor
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    # Turn image into a torch variable
    input = Variable(image_tensor)
    # Transfer image to CPU (Safety Step)
    input = input.to('cpu')
    # Run the image through the model
    output = model(input)
    # Calculate chances of cancer and return
    chance = np.round(output.data.cpu().numpy()[0][0]*100, 2)
    return chance