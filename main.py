import argparse
import gc

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchinfo import summary

import image_dataset
import training
import utils
import wandb
from architectures import resnet50, inception_v3, efficientnet_b0, cornet_z, cornet_s, vit, resnet18

wandb.login()


def adjust_learning_rate(optimizer, epoch, learning_rate):

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f"LR: {lr}")


def main():
    ###################################################################################################################
    # Parse user input
    ###################################################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', required=True,
                        choices=['resnet18', 'resnet50', 'inception', 'efficientnet', 'cornet_z', 'cornet_s', 'vision_transformer'])
    parser.add_argument('--attention', choices=['cbam', 'aacn', 'none'], default='none')
    parser.add_argument('--blended', action='store_true')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--lr', type=float, default=0.1)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    arch = args.arch.lower()
    # img size for all architectures but inception is 224x224
    img_size = 224
    learning_rate = args.lr

    if arch == 'resnet50':
        model = resnet50.ResNet50(attention=args.attention).get_model().to(device)
    elif arch == 'resnet18':
        model = resnet18.ResNet18(attention=args.attention).get_model().to(device)
    elif arch == 'inception':
        model = inception_v3.Inception().get_model().to(device)
        img_size = 299
    elif arch == 'efficientnet':
        model = efficientnet_b0.EfficientNetB0().get_model().to(device)
    elif arch == 'cornet_z':
        model = cornet_z.CORnet(attention=args.attention).get_model().to(device)
    elif arch == 'cornet_s':
        model = cornet_s.CORnet().get_model().to(device)
    elif arch == 'vision_transformer':
        model = vit.ViT().get_model().to(device)

    # learning parameters
    epochs = args.epochs
    batch_size = args.batch_size
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-5, momentum=0.9)
    criterion = torch.nn.BCELoss()  # Binary Cross Entropy

    summary(model, input_size=(batch_size, 3, 224, 224), depth=3, verbose=1)

    if args.blended:
        if args.cbam:
            save_name = f"{arch}_cbam_blended"
        elif args.aacn:
            save_name = f"{arch}_aacm_blended"
        else:
            save_name = f"{arch}_blended"
        base = "../blended_dataset/train/"
        csv_base = "../csv/blended/"
    else:
        if args.attention == 'cbam':
            save_name = f"{arch}_cbam"
        elif args.attention == 'aacn':
            save_name = f"{arch}_aacm"
        else:
            save_name = arch
        base = "../dataset/train/"
        csv_base = "../csv/"

    wandb.init(project=arch,
               config={'lr': learning_rate, 'batch_size': batch_size, 'n_epochs': epochs, 'optimizer': 'RMSprop',
                       'wandb_nb': 'wandb_three_in_one_hm'})

    wandb.watch(model)

    ###################################################################################################################
    # Prepare data for training
    ###################################################################################################################

    train_csv = pd.read_csv(f'{csv_base}train.csv')
    train_set = image_dataset.ImageDataset(train_csv, base, 'train', img_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)

    val_csv = pd.read_csv(f'{csv_base}val.csv')
    valid_set = image_dataset.ImageDataset(val_csv, base, 'val', img_size)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    ###################################################################################################################
    # Start training
    ###################################################################################################################

    train_loss = []
    valid_loss = []

    for epoch in range(epochs):
        print(f"********************* EPOCH {epoch + 1} OF {epochs} *********************")

        adjust_learning_rate(optimizer, epoch, learning_rate)

        train_epoch_loss = training.train(model, arch, train_loader, optimizer, criterion, train_set, device)
        valid_epoch_loss = training.validate(model, arch, valid_loader, criterion, valid_set, device)

        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        print("{:<20} {:<15}".format("Train epoch loss:", f"{train_epoch_loss:.4f}"))
        print("{:<20} {:<15}".format("Validation epoch loss:", f"{valid_epoch_loss:.4f}"))

        utils.log_loss(save_name, epoch + 1, train_epoch_loss, valid_epoch_loss)

        if epoch % 30 == 0 and epoch != 0:
            utils.save_checkpoint(False, epoch, learning_rate, batch_size, model.state_dict(),
                                  optimizer.state_dict(), criterion, 8, len(train_set), save_name)
    ###################################################################################################################
    # Save model and loss plot
    ###################################################################################################################
    wandb.run.finish()

    utils.save_checkpoint(True, epochs, learning_rate, batch_size, model.state_dict(), optimizer.state_dict(),
                          criterion, 8, len(train_set), save_name)

    utils.plot_loss(save_name, train_loss, valid_loss)

    del model
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
