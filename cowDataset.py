from torch.utils.data import Dataset
from PIL import Image
import os


class CustomDataset(Dataset):
    def __init__(self, root_dir, image_list, label_map, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = image_list.tolist()
        self.label_map = label_map

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_name = f'{self.root_dir}/{self.image_list[index]}'
        image_split = img_name.split("/")
        cam_id = image_split[-3]
        date_id = image_split[-4]
        label = self.label_map[(date_id, cam_id, image_split[-2])]
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            images = self.transform(image)
        return images, label, cam_id, date_id
