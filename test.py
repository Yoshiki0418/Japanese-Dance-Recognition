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
  test_loader = torch.utils.data.DataLoader(test_set, **loader_args, shuffle=True)

  # -----model-----

  ModelClass = get_model(args.model.name)
  model=ModelClass().to(args.device)
  model.load_state_dict(torch.load('outputs/2024-10-16/20-43-30/model_best.pt',map_location=torch.device('cpu'))) 


  # -----推論-----
  i=0
  predictions_list=[]

  model.eval()

  with torch.no_grad():
      for test in test_loader:
          i=i+1
          test = test.to(args.device)

          predict = model(test)
          predicted_labels = torch.argmax(predict, dim=1).cpu().numpy()
          predictions_list.append(predicted_labels)
          if i==2:
              break


  # -----提出-----
  print(len(predictions_list))
  current_date = datetime.now().strftime("%Y%m%d")
  filename = f"{args.model.name}_submission_{current_date}.csv"

  sample=pd.read_csv('sample_submit.csv')
  sample.iloc[:,1]=predictions_list
  sample.to_csv(filename, index=False)

if __name__ == "__main__":
    run()