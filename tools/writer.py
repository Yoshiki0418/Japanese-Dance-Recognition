import wandb
from omegaconf import OmegaConf, DictConfig

class WandBMetricsWriter():
    def __init__(
        self,
        project_name: str,
        config: dict,
        model_name: str = None,
    ) -> None:
        self.project_name = project_name
        self.name = model_name

        # configがDictConfigなら辞書形式に変換する
        if isinstance(config, DictConfig):
            config_dict = OmegaConf.to_container(config, resolve=True)
        else:
            config_dict = config

        wandb.init(project=project_name,entity="yoshi-ai", name=self.name, config=config_dict)

    def __call__(
            self, 
            epoch: int, 
            train_loss: float, 
            train_acc: float,
            val_loss: float,
            val_acc: float,
        ) -> None:
        wandb.log(
            {"train_loss": train_loss,
             "train_acc": train_acc,
             "val_loss": val_loss,
             "val_acc": val_acc,
             }, step=epoch)

    def finish(self) -> None:
        wandb.finish()