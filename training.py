import torch
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
import wandb
# training function
from architectures.model import Model
from image_dataset import ImageDataset


def train(
        model: Module,
        model_name: str,
        dataloader: DataLoader,
        optimizer: Optimizer,
        criterion: _Loss,
        train_data: ImageDataset,
        device: torch.device,
        log: bool,
) -> float:

    print('Training')
    model.train()

    counter = 0
    train_running_loss = 0.0

    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data) / dataloader.batch_size)):
        counter += 1
        img, target = data['image'].to(device), data['label'].to(device)

        output = model(img)
        optimizer.zero_grad()

        # apply sigmoid activation to get all the outputs between 0 and 1
        # Inception returns an InceptionOutput type while ResNet returns a tensor, hence the differentiation
        if model_name == 'inception':
            output = torch.sigmoid(output.logits)
        else:
            output = torch.sigmoid(output)

        loss = criterion(output, target)

        log_dict = {f"train-loss": loss}
        if log:
            if i % 50 == 0:
                log_dict[f"train_examples"] = [wandb.Image(i) for i in img[:10]]
            wandb.log(log_dict)

        train_running_loss += loss.item()

        # backpropagation
        loss.backward()
        # update optimizer parameters
        optimizer.step()

    train_loss = train_running_loss / counter
    return train_loss


# validation function
def validate(
        model: Module,
        dataloader: DataLoader,
        criterion: _Loss,
        val_data: ImageDataset,
        device: torch.device,
        log: bool
) -> (float, float):

    print('Validating')
    model.eval()
    counter = 0
    val_running_loss = 0.0
    classes = ['boat', 'car', 'cat', 'chair', 'dog', 'face', 'flower', 'plane']
    correct = 0.
    total = 0.
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data) / dataloader.batch_size)):
            counter += 1
            img, target = data['image'].to(device), data['label'].to(device)

            target_indices = [i for i in range(len(target[0])) if target[0][i] == 1]
            output = model(img)

            # apply sigmoid activation to get all the outputs between 0 and 1
            output = torch.sigmoid(output)

            predicted = output.detach().cpu()
            prediction_array = predicted[0].numpy()

            prediction_indices = sorted(range(len(prediction_array)), key=lambda j: prediction_array[j])[-2:]
            res = utils.check_if_both_one_or_none_correct(classes, target_indices, prediction_indices)
            if res == 'both_correct':
                correct += 1

            total += 1

            loss = criterion(output, target)
            val_running_loss += loss.item()

            if log:
                wandb.log({f"val-loss": loss})

        epoch_acc = 100 * correct / total
        if log:
            wandb.log({f'val-accuracy': epoch_acc})
        print(f"Correct: {correct}, total: {total}, validation accuracy: {epoch_acc:.4f}")

        val_loss = val_running_loss / counter

        return val_loss, epoch_acc
