import numpy as np
import pandas as pd
import torch
import hydra
from datetime import datetime
from omegaconf import DictConfig
from torchvision import transforms
from src.models import get_model
from src.data.get_path import get_image_paths_and_labels
from src.data.datasets import Image_Dataset

@hydra.main(version_base=None,config_path='configs',config_name='config')
def run(args: DictConfig):

  # -----dataLoader-----
  loader_args = {"batch_size": args.model.batch_size, "num_workers": args.num_workers}
  test_path=get_image_paths_and_labels('test')

  transform = transforms.Compose([
      transforms.Resize((224, 224)),  
      transforms.ToTensor()     
  ])
  test_set = Image_Dataset("test", test_path, transform=transform)
  test_loader = torch.utils.data.DataLoader(test_set)

  # -----model-----

  ModelClass = get_model(args.model.name)
  model=ModelClass().to(args.device)
  model.load_state_dict(torch.load('/content/drive/MyDrive/Deep_Learning/Japanese-Dance-Recognition/outputs/2024-10-13/07-33-36/model_best.pt',map_location=torch.device('cpu'))) 


  # -----推論-----
  predictions_list=[]
  image_paths_list = []

  model.eval()

  with torch.no_grad():
      for test_batch in test_loader:
          images, image_paths = test_batch
          images = images.to(args.device)

          predict = model(images)
          predicted_labels = torch.argmax(predict, dim=1).cpu().numpy()

          predictions_list.extend(predicted_labels)
          image_paths_list.extend(image_paths)

  # -----提出-----
  current_date = datetime.now().strftime("%Y%m%d")
  filename = f"{args.model.name}_submission_{current_date}.csv"

  # CSVファイル作成
  submission_df = pd.DataFrame(list(zip(image_paths_list, predictions_list))) 
  
  submission_df.to_csv(filename, index=False, header=False) 
  print(f"Submission file saved as {filename}")

if __name__ == "__main__":
    run()