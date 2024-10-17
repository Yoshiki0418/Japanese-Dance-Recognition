import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from typing import Optional
from pathlib import Path

def calculate_confusion_matrix(
        y_true: list[str | float | int], 
        y_pred: list[str | float | int], 
        save_path: Path,
        labels: Optional[list[str | float | int]] = None
    ) -> np.ndarray:
    """
    実際のラベルと予測されたラベルから混同行列を計算する関数。

    Args:
    - y_true (list or np.ndarray): 実際のラベル
    - y_pred (list or np.ndarray): 予測されたラベル
    - labels (list, optional): クラスのリスト

    Returns:
    - np.ndarray: 混同行列
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plot_confusion_matrix(cm, labels, save_path)
    
    return cm

def plot_confusion_matrix(
        cm: np.ndarray, 
        labels,
        save_path: Path, 
        title="Confusion Matrix", 
        cmap='Blues'
    ) -> None:
    """
    混同行列を可視化するし、結果を保存する関数

    Args:
    - cm (np.ndarray): 混同行列
    - labels (list): クラスのラベル
    - title (str, optional): タイトル
    - cmap (str, optional): カラーマップ
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.show()

"""
# 使用例
y_true = [0, 1, 2, 2, 0, 1, 0]
y_pred = [0, 0, 2, 2, 0, 2, 1]
labels = [0, 1, 2]

# 混同行列の計算
cm = calculate_confusion_matrix(y_true, y_pred, labels)

# 混同行列の可視化
plot_confusion_matrix(cm, labels, title="Confusion Matrix Example")
"""