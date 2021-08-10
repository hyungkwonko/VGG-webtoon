import os
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.datasets as datasets
from torchvision import transforms

input_size = 224

data_transforms = {
    # 'train': transforms.Compose([
    #     transforms.RandomResizedCrop(input_size),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ]),
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class Webtoon_Data(datasets.ImageFolder):

    def __init__(self, root='data', split='train'):
        super(Webtoon_Data, self).__init__(root)

        assert os.path.exists(root), "root: {} not found.".format(root)

        self.root = root
        self.split = split
        if self.split == 'train':
            self.labels_loc = os.path.join(root, 'webtoon_process_train.csv')
            # self.labels_loc = os.path.join(root, 'webtoon_process_train_sample.csv')
            self.transform = data_transforms['train']
        else:
            self.labels_loc = os.path.join(root, 'webtoon_process_val.csv')
            # self.labels_loc = os.path.join(root, 'webtoon_process_val_sample.csv')
            self.transform = data_transforms['val']
        self.img_labels = pd.read_csv(self.labels_loc)  
        self.paths = np.array(self.img_labels['path'])
        self.labels = np.array(self.img_labels['label'])
        self.num = len(self.paths)

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        path = os.path.join(self.root, self.paths[index])
        label = self.labels[index]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        out = {
            'image': image,
            'label': label,
            # 'index': index
            }

        return out