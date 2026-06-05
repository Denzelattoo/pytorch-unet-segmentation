import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2 

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size=(256, 256)):
        super().__init__()
        self.image_paths = image_paths
        self.mask_paths = mask_paths

        self.img_transform = v2.Compose([
            v2.Resize(img_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

        self.mask_transform = v2.Compose([
            v2.Resize(img_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True) 
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image_tensor = torch.tensor(self.img_transform(image), dtype=torch.float32)
        mask_tensor = torch.tensor(self.mask_transform(mask), dtype=torch.float32)
        
        return image_tensor, mask_tensor
