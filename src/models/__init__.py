from .standard import *
from .transfer_learning import *
from .custom import *
from .common import *
from .utils import *

def get_model(model_name: str):
    models = {
        "Handfan_VGG16": Handfan_VGG16,
    }
    try:
        return models[model_name]
    except KeyError:
        raise ValueError(f"Unknown model name: {model_name}")
