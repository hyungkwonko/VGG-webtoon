import time
import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from data.webtoon_load import Webtoon_Data
from torchvision import models
from tqdm import tqdm
import logging
from datetime import datetime


num_workers = 8
num_epochs = 100
batch_size = 128
use_pretrained = True
num_classes = 425  # number of webtoons
model_name = 'vgg16_bn' # vgg16
weight_decay = 1e-5 # 1e-5
learning_rate = 1e-3


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for batch in tqdm(dataloaders[phase]):
                inputs = batch['image'].to(device)
                labels = batch['label'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, f'model/{model_name}_webtoon_{batch_size}_{learning_rate}_{weight_decay}.pth')
            if phase == 'val':
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


if __name__ == '__main__':

    os.makedirs('log', exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d,%H:%M:%S',
        handlers=[
            logging.FileHandler(os.path.join('log', f'{model_name}_finetune_{datetime.now().time()}.log')),
            logging.StreamHandler()
        ]
    )

    logging.info(f'args info...')
    logging.info(f'batch_size: {batch_size}')
    logging.info(f'model_name: {model_name}')
    logging.info(f'weight_decay: {weight_decay}')
    logging.info(f'learning_rate: {learning_rate}')

    logging.info('Initializing Datasets and Data_loader...')

    webtoon_datasets = {
        'train': Webtoon_Data(root=os.path.join('data'), split='train'),
        'val': Webtoon_Data(root=os.path.join('data'), split='val')
        }

    dataloaders_dict = {
        'train': utils.data.DataLoader(webtoon_datasets['train'], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'val': utils.data.DataLoader(webtoon_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers),    
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info('init models...')

    if model_name == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=True, progress=True)
    else:
        model = models.vgg16(pretrained=use_pretrained)
    model.classifier[6] = nn.Linear(4096, num_classes)
    model = model.to(device)

    # if torch.cuda.device_count() > 1:
    #     logging.info(f"number of devices...: {torch.cuda.device_count()}")
    #     model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # l2 norm
    criterion = nn.CrossEntropyLoss()
    model, _ = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)