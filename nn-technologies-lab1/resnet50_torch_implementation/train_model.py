import time
import torch
from tqdm.autonotebook import tqdm, trange


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
    return model, losses, f1_macro, best_model_wts, best_epoch
