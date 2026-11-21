"""Independent AnimalAct 2.0 Proposal models and release-v1 clip input."""

from .dataset import ReleaseClipDataset
from .collate import collate_clips, move_batch_to_device
from .backbone import ResNet18Backbone
from .model1 import Model1
from .model2 import Model2
from .model3 import Model3

__all__ = ["ReleaseClipDataset", "collate_clips", "move_batch_to_device", "ResNet18Backbone", "Model1", "Model2", "Model3"]
