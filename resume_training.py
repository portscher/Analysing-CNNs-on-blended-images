import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

import image_dataset
import utils
from architectures import resnet50, inception, efficientnet, cornet_z
from training import train, validate


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 25 epochs"""
    lr = args.lr * (0.1 ** (epoch // 25))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', choices=['resnet', 'inception', 'efficientnet', 'cornet'], required=True)
    parser.add_argument('--path', required=True)
    parser.add_argument('--new_epochs', type=int, default=10, required=True)
    parser.add_argument('--blended', action='store_true')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()

    # learning parameters
    batch_size = args.batch
    new_epochs = args.new_epochs
    lr = args.lr
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the trained model
    arch = args.arch
    path = args.path
    img_size = 224
    if arch == 'resnet':
        model = resnet50.ResNet50(path=path, train_from_scratch=False).get_model().to(device)
    elif arch == 'inception':
        model = inception.Inception(path=path, train_from_scratch=False).get_model().to(device)
        img_size = 299
    elif arch == 'efficientnet':
        model = efficientnet.EfficientNetB0(path=path, train_from_scratch=False).get_model().to(device)
    elif arch == 'cornet':
        model = cornet_z.CORnet(path=path, train_from_scratch=False).get_model().to(device)

    # initialize optimizer  before loading optimizer state_dict
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0001, momentum=0.9)

    checkpoint = torch.load(args.path)
    # load model weights state_dict
    model.load_state_dict(checkpoint['state_dict'])
    # load trained optimizer state_dict
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochs = checkpoint['epoch']
    # load the criterion
    criterion = checkpoint['loss']
    print(f"Previously trained for {epochs} epochs.")
    # train for more epochs
    print(f"Train for {new_epochs} more epochs.")

    if args.blended:
        save_name = f"{args.arch}_blended"
        training_folder = "../blended_dataset/train/"
        validation_folder = "../blended_dataset/val/"
    else:
        save_name = args.arch
        training_folder = "../reg_dataset/train/"
        validation_folder = "../reg_dataset/val/"

    train_csv = pd.read_csv('../csv/train_labels.csv')
    train_set = image_dataset.ImageDataset(train_csv, training_folder, 'train', img_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    val_csv = pd.read_csv('../csv/val_labels.csv')
    valid_set = image_dataset.ImageDataset(val_csv, validation_folder, 'val', img_size)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    train_loss = []
    valid_loss = []

    for epoch in range(new_epochs):
        print(f"Epoch {epochs + epoch + 1} of {new_epochs + epochs}")

        if epoch == 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001

        if epoch == 20:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001

        train_epoch_loss = train(model, arch, train_loader, optimizer, criterion, train_set, device)
        val_epoch_loss = validate(model, valid_loader, criterion, valid_set, device)
        train_loss.append(train_epoch_loss)
        valid_loss.append(val_epoch_loss)

        utils.log_loss(arch, epochs + epoch, train_epoch_loss, val_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f'Val Loss: {val_epoch_loss:.4f}')

    utils.save_checkpoint(False, new_epochs + epochs + 1, lr, batch_size, model.state_dict(), optimizer.state_dict(),
                          criterion, 8, len(train_set), f'{save_name}_resumed')


if __name__ == '__main__':
    main()
