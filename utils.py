import collections
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import torch


def log_loss(model_name, epoch, train_epoch_loss, valid_epoch_loss):
    df = pd.DataFrame({'epoch': epoch, 'train': train_epoch_loss, 'val': valid_epoch_loss}, index=[0])
    FILENAME = f'{model_name}_loss.csv'
    # if file does not already exist, write header
    if not os.path.isfile(FILENAME):
        df.to_csv(FILENAME)
    else:  # else it exists, append to existing file without header
        df.to_csv(FILENAME, mode='a', header=False)


def plot_loss(model_name, train_loss, valid_loss):
    """
    Plots the loss per epoch of a trained model
    """
    plt.style.use('ggplot')
    # plot the train and validation line graphs
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(valid_loss, color='red', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'outputs/{model_name}_loss.png')


def save_checkpoint(final, epoch, lr, batch_size, model_state_dict, opt_state_dict, criterion, nclasses, ntrain_imgs,
                    save_name):
    """
    Saves a checkpoint/final model to the disk
    """
    checkpoint_dict = {
        'epoch': epoch,
        'learning_rate': lr,
        'batch_size': batch_size,
        'state_dict': model_state_dict,
        'optimizer_state_dict': opt_state_dict,
        'loss': criterion,
        'num_classes': nclasses,
        'num_train_imgs': ntrain_imgs
    }

    # save the trained model to disk
    time = datetime.now().strftime("%b%d")
    if not final:
        torch.save(checkpoint_dict, f"outputs/{save_name}_{time}_epoch{epoch}.pth")
    else:
        torch.save(checkpoint_dict, f"outputs/{save_name}_{time}_final.pth")


def check_if_one_common_element(list1, list2):
    """
    Compares two lists and checks if they have exactly one element in common
    (For documenting prediction accuracy)
    """
    count = 0
    for e in list1:
        for f in list2:
            if e == f:
                count += 1
    return True if count == 1 else False


def check_if_both_one_or_none_correct(classes, target_indices, prediction_indices):
    """
    Compares target and prediction indices and computes whether both predictions,
    one of them or none of them are correct.
    """
    if len(target_indices) == 1:
        if classes[prediction_indices[1]] is classes[target_indices[0]]:
            return 'both_correct'
        else:
            return 'incorrect'
    elif len(target_indices) == 2:
        if collections.Counter(prediction_indices) == collections.Counter(target_indices):
            return 'both_correct'
        elif check_if_one_common_element(prediction_indices, target_indices):
            return 'one_correct'
        else:
            return 'incorrect'


def which_correct(classes, target_indices, prediction_indices):
    common_element = list(set(target_indices).intersection(prediction_indices))[0]
    cl = classes[common_element]

    return hyperclasses(cl)


def print_results(both_correct, one_correct, none_correct, hyperclass_dict):
    """
    Prints the results of the prediction
    """

    total = both_correct + one_correct + none_correct
    percentage_both = '{:.2%}'.format(both_correct / total)
    percentage_one = '{:.2%}'.format(one_correct / total)
    percentage_none = '{:.2%}'.format(none_correct / total)

    accuracy_dict = {"Both correct": [both_correct, percentage_both],
                     "One correct": [one_correct, percentage_one],
                     "None correct": [none_correct, percentage_none]}

    for k, v in accuracy_dict.items():
        total, percentage = v
        print("{:<20} {:<15} {:<15}".format(k, total, percentage))
    print("\n")
    print("{:<20} {:<15} {:<15} {:<15} {:<15}".format('Combination', 'Both correct', 'One correct', 'Ratio', 'None correct'))
    for k, v in hyperclass_dict.items():
        both, one, none, ratio = v
        new_total = both + one + none
        both = '{:.2%}'.format(both / new_total)
        one = '{:.2%}'.format(one / new_total)
        none = '{:.2%}'.format(none / new_total)

        # calculate the ratio in percent
        tmp_a = ratio[0]
        tmp_b = ratio[1]
        tmp_tot = tmp_a + tmp_b
        ratio = ('{:.2%}'.format(tmp_a/tmp_tot), '{:.2%}'.format(tmp_b/tmp_tot))
        ratio = '{0}'.format(ratio)

        print("{:<20} {:<15} {:<13} {:<25} {:<15}".format(k, both, one, ratio, none))


def add_results_to_experiment_table(model_name, num_classes, num_training_imgs, num_test_imgs, both_correct,
                                    one_correct, none_correct, epochs, learning_rate, batch_size):
    """
    Every time a model is tested this function saves the model information and prediction accuracy to a csv file.
    """
    now = datetime.now()
    dt_string = now.strftime('%d/%m/%Y %H:%M')

    total = both_correct + one_correct + none_correct
    percentage_both = '{:.2%}'.format(both_correct / total)
    percentage_one = '{:.2%}'.format(one_correct / total)
    percentage_none = '{:.2%}'.format(none_correct / total)

    df = pd.DataFrame({'date': dt_string,
                       'model': model_name,
                       'epochs': epochs,
                       'batch size': batch_size,
                       'learning rate': learning_rate,
                       '#classes': num_classes,
                       '#training images': num_training_imgs,
                       '#test images': num_test_imgs,
                       'both correct': percentage_both,
                       'one correct': percentage_one,
                       'none correct': percentage_none}, index=[0])

    FILENAME = 'experiments.csv'

    if not os.path.isfile(FILENAME):
        # if file does not already exist, write header
        df.to_csv(FILENAME)
    else:
        # append to existing file without header
        df.to_csv(FILENAME, mode='a', header=False)


class HyperClass:
    human = "Human"
    nature = "Nature"
    man_made = "ManMade"


def hyperclasses(imgclass):
    MAN_MADE = ['boat', 'car', 'plane', 'chair']
    NATURE = ['flower', 'cat', 'dog']
    if imgclass in MAN_MADE:
        # boat, car, chair, plane
        return HyperClass.man_made
    elif imgclass in NATURE:
        # flower, cat, dog
        return HyperClass.nature
    else:
        # faces
        return HyperClass.human


def hyperclass_comb(class1, class2):
    hyperclass_list = [hyperclasses(class1), hyperclasses(class2)]
    hyperclass_list.sort()
    return hyperclass_list


def update_hyperclass_dict(hyperclass_dict, res, which_class, class1, class2):
    hyperclass_list = hyperclass_comb(class1, class2)
    dict_entry_name = f"{hyperclass_list[0]}-{hyperclass_list[1]}"

    tmp_list = hyperclass_dict[dict_entry_name]
    fst, snd = tmp_list[3]
    if res == 'both_correct':
        tmp_list[0] += 1
    elif res == 'one_correct':
        tmp_list[1] += 1
        if hyperclass_list[0] == which_class:
            fst += 1
        else:
            snd += 1
        tmp_list[3] = (fst, snd)
    else:
        tmp_list[2] += 1

    hyperclass_dict.update([(dict_entry_name, tmp_list)])
    return hyperclass_dict
