from .custom import Handfan_VGG16, Handfan_ViT


def get_model(model_name: str):
    models = {
        "Handfan_VGG16": Handfan_VGG16,
        "Handfan_ViT": Handfan_ViT,
    }
    try:
        return models[model_name]
    except KeyError:
        raise ValueError(f"Unknown model name: {model_name}")
