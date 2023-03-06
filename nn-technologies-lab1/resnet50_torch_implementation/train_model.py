import time
import pandas as pd
import torch
import torch.nn as  nn
import torch.nn.functional as F
import torchvision
from tqdm.autonotebook import tqdm, trange
from torchvision.transforms import transforms
from torchmetrics import F1Score
from torchinfo import summary
import matplotlib.pyplot as plt


train_data = torchvision.datasets.StanfordCars(root="./data", split="train",
                                               download=True,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Resize((224,224))
                                               ]))
loader = torch.utils.data.DataLoader(train_data, batch_size= 64, shuffle=False, num_workers=1)

from torchvision.transforms.functional import rotate

# For stanford cars dataset
mean = torch.tensor([0.4707, 0.4602, 0.4550])
std = torch.tensor([0.2638, 0.2629, 0.2678])

train_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize((224, 224)),
                                      transforms.Normalize(mean, std),
                                      transforms.RandomAdjustSharpness(sharpness_factor=2),
                                      transforms.RandomAutocontrast(),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomAffine(degrees=35, translate=(0.3, 0.3)),
                                      ])

test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize((224, 224)),
                                     transforms.Normalize(mean, std)])

train_data = torchvision.datasets.StanfordCars(root="./data",
                                           split="train",
                                           download=True,
                                           transform=train_transform)

val_data = torchvision.datasets.StanfordCars(root="./data",
                                         split="train",
                                         download=True,
                                         transform=test_transform)

val_count = round(len(train_data) * 0.2)
train_count = len(train_data) - val_count

train_data, _ = torch.utils.data.random_split(train_data, [train_count, val_count])
_, val_data = torch.utils.data.random_split(val_data, [train_count, val_count])


test_data = torchvision.datasets.StanfordCars(root="./data",
                                          split="test",
                                          download=True,
                                          transform = test_transform)

dataset_sizes = {'train': len(train_data),
                 'val': len(val_data),
                 'test': len(test_data)}

print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

classes = train_data.dataset.classes
batch_size = 64  # TODO: add automatic batch size finder for PL

train_dataloader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2,
                                               pin_memory=True)


val_dataloader = torch.utils.data.DataLoader(val_data,
                                             batch_size = batch_size,
                                             shuffle = False,
                                             num_workers = 2)


test_dataloader = torch.utils.data.DataLoader(test_data,
                                             batch_size = batch_size,
                                             shuffle = False,
                                             num_workers = 2)

img_size = next(iter(train_dataloader))[0].shape[2]

print(f'Количество классов в датасете: {len(classes)}')
print(f'Размерность одного батча: {next(iter(train_dataloader))[0].shape}')

dataloaders = {'train': train_dataloader,
               'val': val_dataloader,
               'test': test_dataloader}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batch_norm3(x)

        # downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        # add identity
        x += identity
        x = self.relu(x)

        return x


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm2(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        print(x.shape)
        print(identity.shape)
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)


def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, channels)


def ResNet101(num_classes, channels=3):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, channels)


def ResNet152(num_classes, channels=3):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, channels)

model = ResNet50(len(classes))
summary(model, input_size=(batch_size, 3, img_size, img_size))


def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()
    best_model_wts = model.state_dict()
    best_f1 = 0.0
    best_loss = 1.0
    losses = {'train': [], 'val': []}
    f1_macro = {'train': [], 'val': []}
    pbar = trange(num_epochs, desc='Epoch:')

    for epoch in pbar:

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            true_labels = []
            pred_labels = []

            for data in dataloaders[phase]:
                inputs, labels = data
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                else:
                    inputs, labels = inputs, labels

                if phase == 'train':
                    optimizer.zero_grad()

                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                preds = torch.argmax(outputs, -1)
                loss = criterion(outputs, labels)
                true_labels += labels.tolist()
                pred_labels += outputs.tolist()

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / dataset_sizes[phase]
            f1_score = f1(torch.tensor(pred_labels), torch.tensor(true_labels))
            losses[phase].append(epoch_loss)
            f1_macro[phase].append(f1_score)

            pbar.set_description('{} Loss: {:.4f} F1: {:.4f}'.format(
                phase, epoch_loss, f1_score
            ))

            if phase == 'val' and f1_score > best_f1:
                best_f1 = f1_score
                if epoch_loss < best_loss:
                    best_model_wts = model.state_dict()
                    best_epoch = epoch
                    best_loss = epoch_loss


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val f1_macro: {:4f}'.format(best_f1))

    model.load_state_dict(best_model_wts)
    return model, losses, f1_macro, best_model_wts , best_epoch


use_gpu = torch.cuda.is_available()

if use_gpu:
    model = model.cuda()

lr = 0.001
f1 = F1Score(task='multiclass', num_classes=len(classes), average="macro")
loss = nn.CrossEntropyLoss()
# resnet_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# resnet_model, resnet_losses, resnet_f1_macro, resnet_weights, best_epoch = train_model(model, loss, resnet_optimizer, num_epochs=120)
#
# metrics = pd.DataFrame({'train_loss': resnet_losses['train'],
#                         'val_loss': resnet_losses['val'],
#                         'train_f1': resnet_f1_macro['train'],
#                         'val_f1': resnet_f1_macro['val']})
#
# print('------------------ ResNet statistics ------------------')
# print(metrics)
# torch.save(resnet_weights, f'weights/resnet_weights_adam_epoch_{best_epoch+1}.h5')
# metrics.to_csv('logs/resnet_50_adam_optimizer')

model = ResNet50(len(classes))
if use_gpu:
    model = model.cuda()
resnet_optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
resnet_model, resnet_losses, resnet_f1_macro, resnet_weights, best_epoch = train_model(model, loss, resnet_optimizer, num_epochs=120)

metrics = pd.DataFrame({'train_loss': resnet_losses['train'],
                         'val_loss': resnet_losses['val'],
                         'train_f1': resnet_f1_macro['train'],
                         'val_f1': resnet_f1_macro['val']})

print('------------------ ResNet statistics ------------------')
print(metrics)
torch.save(resnet_weights, f'weights/resnet_weights_adamax_epoch_{best_epoch+1}.h5')
metrics.to_csv('logs/resnet_50_adamax_optimizer')
