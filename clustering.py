import os
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from glob import glob
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import pandas as pd


input_size = 224
use_pretrained = False
num_classes = 425  # number of webtoons
# model_path = 'model/vgg16_bn_webtoon_128_0.001_1e-07.pth'
# model_path = 'model/vgg16_bn_webtoon_128_0.0001_1e-05_0.9.pth'
model_path = 'model/vgg16_bn_webtoon_64_0.0001_1e-05_0.98.pth'
number_of_clusters = 425
save_imgs = True
model_name = 'vgg16_bn' # vgg16
clustering_method = 'kmeans'

data_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


if __name__ == '__main__':

    os.makedirs('result', exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('init models...')

    if model_name == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=use_pretrained)
    else:
        model = models.vgg16(pretrained=use_pretrained)
    model.classifier[6] = nn.Linear(4096, num_classes)

    print('load models...')

    model.load_state_dict(torch.load(model_path), strict=False)
    model = model.to(device)
    model.eval()

    # new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    # model.classifier = new_classifier

    files = glob('data/webtoon_test/*.jpg')

    print("load image files...")

    features = []
    num_images = len(files)

    for file in tqdm(files):
        image = Image.open(file).convert("RGB")
        image = data_transforms(image)
        image = image.unsqueeze(0).to(device)
        feature = model(image).cpu().detach().squeeze().tolist()
        features.append(feature)

    features = np.array(features)

    print(f"features.shape: {features.shape}")

    if clustering_method == 'dbscan':
        clustering = DBSCAN(eps=100, min_samples=10, n_jobs=-1).fit(features)
    else:
        clustering = KMeans(n_clusters=number_of_clusters, random_state=0).fit(features)
    labels = clustering.labels_
    print(f"{clustering_method} clustering finished...")
    print(f"unique: {np.unique(labels, return_counts=True)}...")

    save_file = pd.DataFrame()
    save_file['seed'] = np.arange(num_images)
    save_file['cluster'] = labels
    save_file.to_csv('result/cluster_info.csv', index=False)

    print("csv file saved successfully...")

    if save_imgs:
        for i in set(labels):
            os.makedirs(f'result/{i}', exist_ok=True)

        for label, file in zip(labels, files):
            image = Image.open(file).convert("RGB")
            file = file.replace('data/webtoon_test/', f'result/{label}/')
            image.save(file)

        print("all image files saved successfully...")


