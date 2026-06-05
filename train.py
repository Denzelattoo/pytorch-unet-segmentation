import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import SegmentationDataset
from src.model import Unet

BATCH_SIZE = 4
LR = 0.001
EPOCHS = 10

train_image_paths = ['path/to/train/image1.jpg', 'path/to/train/image2.jpg', 'path/to/train/image3.jpg']
train_mask_paths = ['path/to/train/mask1.png', 'path/to/train/mask2.png', 'path/to/train/mask3.png']

model = Unet(num_classes = 1)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LR)


train_dataset = SegmentationDataset(image_paths = train_image_paths, mask_paths = train_mask_paths)

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)

print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for images, masks in train_loader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks.unsqueeze(1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader) 
    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}")