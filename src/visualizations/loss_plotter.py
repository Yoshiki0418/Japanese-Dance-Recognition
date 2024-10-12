import matplotlib.pyplot as plt

def loss_curve_plotter(
        train_loss_history: list,
        val_loss_history: list,
        save_path: str,
    ) -> None:
    plt.figure(figsize=[10,8], dpi=200)
    plt.plot(train_loss_history, "r", label="train loss")
    plt.plot(val_loss_history, "b", label="val loss")
    plt.legend()
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.title("Loss Curves", fontsize=18)

    plt.savefig(save_path)
    plt.show()