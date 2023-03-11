import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
from torchmetrics import F1Score
from torchinfo import summary
from resnet_model import *
from train_model import train_model

train_data = torchvision.datasets.StanfordCars(root="./data", split="train",
                                               download=True,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Resize((224, 224))
                                               ]))
loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=False, num_workers=1)

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
                                              transform=test_transform)

dataset_sizes = {'train': len(train_data),
                 'val': len(val_data),
                 'test': len(test_data)}

print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

classes = train_data.dataset.classes
batch_size = 64

train_dataloader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2,
                                               pin_memory=True)

val_dataloader = torch.utils.data.DataLoader(val_data,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=2)

test_dataloader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=2)

img_size = next(iter(train_dataloader))[0].shape[2]

print(f'Количество классов в датасете: {len(classes)}')
print(f'Размерность одного батча: {next(iter(train_dataloader))[0].shape}')

dataloaders = {'train': train_dataloader,
               'val': val_dataloader,
               'test': test_dataloader}

model = ResNet50(len(classes))
summary(model, input_size=(batch_size, 3, img_size, img_size))

use_gpu = torch.cuda.is_available()

if use_gpu:
    model = model.cuda()

lr = 0.001
f1 = F1Score(task='multiclass', num_classes=len(classes), average="macro")
loss = nn.CrossEntropyLoss()
# resnet_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# resnet_model, resnet_losses, resnet_f1_macro, resnet_weights, best_epoch = train_model(model,
#                                                                                        loss,
#                                                                                        resnet_optimizer,
#                                                                                        num_epochs=120)
#
# metrics = pd.DataFrame({'train_loss': resnet_losses['train'],
#                         'val_loss': resnet_losses['val'],
#                         'train_f1': resnet_f1_macro['train'],
#                         'val_f1': resnet_f1_macro['val']})
#
# print('------------------ ResNet statistics ------------------')
# print(metrics)
# torch.save(resnet_weights, f'weights/resnet_weights_adam_epoch_{best_epoch+fold1}.h5')
# metrics.to_csv('logs/resnet_50_adam_optimizer')

model = ResNet50(len(classes))
if use_gpu:
    model = model.cuda()
resnet_optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
resnet_model, resnet_losses, resnet_f1_macro, resnet_weights, best_epoch = train_model(model,
                                                                                       loss,
                                                                                       resnet_optimizer,
                                                                                       num_epochs=120)

metrics = pd.DataFrame({'train_loss': resnet_losses['train'],
                        'val_loss': resnet_losses['val'],
                        'train_f1': resnet_f1_macro['train'],
                        'val_f1': resnet_f1_macro['val']})

print('------------------ ResNet statistics ------------------')
print(metrics)
torch.save(resnet_weights, f'weights/resnet_weights_adamax_epoch_{best_epoch + 1}.h5')
metrics.to_csv('logs/resnet_50_adamax_optimizer')
