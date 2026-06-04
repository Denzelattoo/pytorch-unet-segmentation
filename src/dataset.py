import torch
from torch.utils.data import Dataset
from PIL import Image

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size = (256, 256)):
        super().__init__()
        self.image_paths = image_paths
        self.mask_paths = mask_paths

        self.img_transform = v2.Compose([
            v2.Resize(img_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale = True)
        ])

        self.mask_transform = v2.Compose([
            v2.Resize(img_size),
            v2.ToImage(),
            v2.ToDtype(torch.long scale = False)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = self.image_paths[index]
        mask = self.mask.paths[index]

        image = Image.open(image_path).convert('RGB')
        mask = Mask.open(mask_paths).convert('L')

        image_tensor = self.img_transform(image)
        mask_tensor = self.mask_transform(mask)
        return image_tensor, mask_tensor
