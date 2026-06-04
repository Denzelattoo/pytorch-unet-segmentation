import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import SegmentationDataset
from src.model import Unet

BATCH_SIZE = 4
LR = 0.001
EPOCHS = 10



train_dataset = SegmentationDataset(image_paths = train_image_paths, mask_paths = train_mask_paths)

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)