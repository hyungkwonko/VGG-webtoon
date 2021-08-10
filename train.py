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

num_workers = 8
num_epochs = 500
batch_size = 64
# batch_size = 16
use_pretrained = True
num_classes = 425  # number of webtoons


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print("model init saved...")
    torch.save(best_model_wts, f'model/vgg_webtoon_{batch_size}.pth')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

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

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, f'model/vgg_webtoon_{batch_size}.pth')
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


if __name__ == '__main__':
        
    print("Initializing Datasets and Dataloaders...")

    webtoon_datasets = {
        'train': Webtoon_Data(root=os.path.join('data'), split='train'),
        'val': Webtoon_Data(root=os.path.join('data'), split='val')
        }

    dataloaders_dict = {
        'train': utils.data.DataLoader(webtoon_datasets['train'], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'val': utils.data.DataLoader(webtoon_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers),    
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('init models...')

    # model = models.vgg16(pretrained=True, progress=True)
    model = models.vgg16_bn(pretrained=use_pretrained)
    model.classifier[6] = nn.Linear(4096, num_classes)

    model = model.to(device)

    print("number of devices...: ", torch.cuda.device_count())

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    model, _ = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)