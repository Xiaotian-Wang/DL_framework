from torch.utils.data import Dataset
from torchvision.io import read_image
import os


class MyDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_name_list = os.listdir(self.img_dir)
        self.img_names_with_labels = {item: 1 if item.startswith('dog') else 0 for item in self.img_name_list}

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        cur_img_dir = os.path.join(self.img_dir, self.img_name_list[idx])
        label = self.img_names_with_labels[self.img_name_list[idx]]
        image = read_image(cur_img_dir)
        if self.transform:
            image = self.transform(image)
        return image, label