"""Only a factory around the unchanged, validated Phase1.5.1 model definitions."""
from animalact2.phase1_5_1 import Model1, Model2, Model3, ResNet18Backbone
from .config import seed_everything


def build_model(config):
    seed_everything(config["seed"])
    backbone = ResNet18Backbone(frozen=True, frame_chunk_size=config["backbone"]["frame_chunk_size"])
    klass = dict(model1=Model1, model2=Model2, model3=Model3)[config["model"]]
    return klass(backbone=backbone)
