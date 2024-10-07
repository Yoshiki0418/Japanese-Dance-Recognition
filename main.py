import os
import pandas as pd
import numpy as np
import mne
import sys
import hydra
from omegaconf import DictConfig
import torch.nn.functional as F
import wandb
from termcolor import cprint
import random
import glob
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchmetrics import Accuracy
from pymatreader import read_mat
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from src.utils import set_seed

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    #--------------------------------
    #          Dataloader
    #--------------------------------
    loader_args = {"batch_size": args.model.batch_size, "num_workers": args.num_workers}

    """メモ
    トレーニングデータと検証データは適当なパスを指定しているだけなので、後で切り替えが必要かも
    チュートリアルでは、データセットをあらかじめ作っている感じでコードを書いているのでチェックしておく
    """
    subject_ids = ['subject0', 'subject1', 'subject2', 'subject3', 'subject4']

    for subject_id in subject_ids:
        print(f"Start training {subject_id}")

        train_dir = os.path.join('test_modeling2', 'train', subject_id)
        val_dir = os.path.join('test_modeling2', 'val', subject_id)

        def transform(array, is_train, seq_length, mask_prob, apply_random, window_size):
            if is_train:
                _, n = array.shape
                s = random.randint(0, n-seq_length)
                ts = array[:, s:s+seq_length]
                ts = add_noise(ts).astype(np.float32)
                
                if apply_random:
                    ts = apply_random_augmentations(ts, mask_prob, window_size)
                else:
                    ts = apply_augmentations_in_order(ts, mask_prob, window_size)

                if random.randint(0, 1):
                    ts_r = ts[:, ::-1].copy()
                    return ts_r              
                return ts
            else:
                ts = array[:,:seq_length].astype(np.float32)
                return ts

        train_set = EEGDataset(
            root=train_dir, 
            seq_length=args.seq_length, 
            is_train=True, 
            transform=transform, 
            use_mixup=args.model.use_mixup, 
            mixup_alpha=args.model.mixup_alpha, 
            augment_factor=args.model.augment_factor,
            mask_prob=args.model.mask_prob,
            apply_random=args.model.apply_random,
            window_size=args.model.window_size,
        )
        train_loader = torch.utils.data.DataLoader(train_set, **loader_args, shuffle=True)

        val_set = EEGDataset(root=val_dir, seq_length=args.seq_length, is_train=False, transform=transform)
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
        max_val_acc = 0
        accuracy = Accuracy(
            task="multiclass", num_classes=train_set.numof_classes
        ).to(args.device)

        subject_model_dir = os.path.join(logdir, f"model_{subject_id}")
        os.makedirs(subject_model_dir, exist_ok=True)

        train_loss_history = []
        val_loss_history = []

        writer = WandBMetricsWriter(project_name = f"{args.project_name}_{subject_id}",
                                     config = args.model,
                                     model_name = args.model.name)

        for epoch in range(args.model.epochs):
            print(f"Epoch {epoch+1}/{args.model.epochs}")

            train_loss, train_acc, val_loss, val_acc = [], [], [], []

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
            
            scheduler.step()

            model.eval()
            y_true_list = []
            y_pred_list = []
            labels = train_set.class_names
            for data in tqdm(val_loader, desc="Validation"):
                X, y = data['seq'].to(args.device), data['label'].to(args.device)

                with torch.no_grad():
                    y_pred = model(X)

                v_loss = F.cross_entropy(y_pred, y)
                val_loss.append(v_loss.item())

                v_acc = accuracy(y_pred, y)
                val_acc.append(v_acc.item())

                y_true_list.append(y.cpu().numpy()) 
                y_pred_list.append(torch.argmax(y_pred, dim=1).cpu().numpy()) 

            y_true_list = np.concatenate(y_true_list).ravel()
            y_pred_list = np.concatenate(y_pred_list).ravel()

            y_true_mapped = [labels[label - 1] for label in y_true_list]       
            y_pred_mapped = [labels[label - 1] for label in y_pred_list]

            print(f"Epoch {epoch+1}/{args.model.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")

            torch.save(model.state_dict(), os.path.join(subject_model_dir, "model_last.pt"))
            if args.use_wandb:
                wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
            
            if np.mean(val_acc) > max_val_acc:
                cprint("New best.", "cyan")
                torch.save(model.state_dict(), os.path.join(subject_model_dir, "model_best.pt"))
                max_val_acc = np.mean(val_acc)
                calculate_confusion_matrix(
                    y_true_mapped,
                    y_pred_mapped, 
                    Path(subject_model_dir) / "best_confusion_matrix.png",
                    labels,
                )

            train_loss_history.append(np.mean(train_loss))
            val_loss_history.append(np.mean(val_loss))

            writer(
                epoch = epoch, 
                train_loss = np.mean(train_loss),
                train_acc = np.mean(train_acc),
                val_loss = np.mean(val_loss),
                val_acc = np.mean(val_acc),
            )

            if epoch % 50 == 0:
                calculate_confusion_matrix(
                    y_true_mapped,
                    y_pred_mapped, 
                    Path(subject_model_dir) / f"epoch{epoch}_confusion_matrix.png",
                    labels,
                )
        
        loss_curve_plotter(train_loss_history, val_loss_history, os.path.join(subject_model_dir, f"{args.model.name}_loss_plotting.png"))
    
        writer.finish()

if __name__ == "__main__":
    run()

