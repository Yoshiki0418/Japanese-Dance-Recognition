import os
import numpy as np
import hydra
from omegaconf import DictConfig
import torch.nn.functional as F
import wandb
from termcolor import cprint
import torch
from tqdm import tqdm
from torchmetrics import Accuracy, F1Score
from sklearn.model_selection import train_test_split
from torchvision import transforms

from src.utils import set_seed
from src.models import get_model
from src.data.datasets import Image_Dataset
from src.data.get_path import get_image_paths_and_labels
from src.visualizations import loss_curve_plotter
from tools import *


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    #--------------------------------
    #          Dataloader
    #--------------------------------
    loader_args = {"batch_size": args.model.batch_size, "num_workers": args.num_workers}

    image_paths, labels = get_image_paths_and_labels("train")


    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    transform = transforms.Compose([
        transforms.Resize((128, 128)),  
        transforms.ToTensor()     
    ])

    train_set = Image_Dataset("train", train_paths, train_labels, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, **loader_args, shuffle=True)

    val_set = Image_Dataset("val", val_paths, val_labels, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, **loader_args, shuffle=False)

    #--------------------------------
    #            Model
    #--------------------------------
    ModelClass = get_model(args.model.name)
    
    model_params = args.model.model_param

    model = ModelClass(train_set.seq_length, **model_params).to(args.device)

    #--------------------------------
    #          Optimizer
    #--------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.model.lr)

    #--------------------------------
    #   Learning Rate Scheduler
    #--------------------------------
    model_scheduler = args.model.scheduler

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **model_scheduler)

    #--------------------------------
    #     Start training
    #--------------------------------
    max_val_f1 = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.numof_classes
    ).to(args.device)

    f1_score = F1Score(
        task="multiclass", num_classes=train_set.numof_classes, average="macro"
    ).to(args.device)


    train_loss_history = []
    val_loss_history = []

    writer = WandBMetricsWriter(project_name = f"{args.project_name}",
                                    config = args.model,
                                    model_name = args.model.name)

    for epoch in range(args.model.epochs):
        print(f"Epoch {epoch+1}/{args.model.epochs}")

        train_loss, train_acc, val_loss, val_acc, train_f1, val_f1 = [], [], [], [], [], []

        model.train()
        for data in tqdm(train_loader, desc="Train"):
            X, y = data['seq'].to(args.device), data['label'].to(args.device)

            y_pred = model(X)

            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

            f1 = f1_score(y_pred, y)
            train_f1.append(f1.item())
        
        scheduler.step()

        model.eval()

        for data in tqdm(val_loader, desc="Validation"):
            X, y = data['seq'].to(args.device), data['label'].to(args.device)

            with torch.no_grad():
                y_pred = model(X)

            v_loss = F.cross_entropy(y_pred, y)
            val_loss.append(v_loss.item())

            v_acc = accuracy(y_pred, y)
            val_acc.append(v_acc.item())

            v_f1 = f1_score(y_pred, y)
            val_f1.append(v_f1.item())

        print(f"Epoch {epoch+1}/{args.model.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | train f1: {np.mean(train_f1):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f} | val f1: {np.mean(val_f1):.3f}")


        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "train_f1": np.mean(train_f1), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc), "val_f1": np.mean(val_f1)})
        
        if np.mean(val_f1) > max_val_f1:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_f1 = np.mean(val_f1)

        train_loss_history.append(np.mean(train_loss))
        val_loss_history.append(np.mean(val_loss))

        writer(
            epoch = epoch, 
            train_loss = np.mean(train_loss),
            train_acc = np.mean(train_acc),
            val_loss = np.mean(val_loss),
            val_acc = np.mean(val_acc),
        )
    
    loss_curve_plotter(train_loss_history, val_loss_history, os.path.join(logdir, f"{args.model.name}_loss_plotting.png"))

    writer.finish()

if __name__ == "__main__":
    run()

