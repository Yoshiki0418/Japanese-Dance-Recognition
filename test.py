import pandas as pd
import torch
from torchvision import transforms
from src.data.get_path import get_image_paths_and_labels
from src.data.datasets import Image_Dataset

# test=pd.read_csv('test')

test_path=get_image_paths_and_labels('test')

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor()     
])
test_set = Image_Dataset("test", test_path, transform=transform)

model_path=''
model = torch.load(model_path)

model.eval()

with torch.no_grad():
  predict = model(test_set)
  pred_numpy = predict.cpu().numpy()

  df = pd.DataFrame(pred_numpy)
  
  
  df.to_csv('predictions.csv', index=False)

