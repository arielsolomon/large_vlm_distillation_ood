from copy import copy
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from trainers.simple_trainer import train, test


def fine_tune_train(net: torch.nn.Module,
                    train_loader_ood: DataLoader, test_loader_ood: DataLoader, test_loader: DataLoader,
                    base_accuracy_on_regular_data: float,
                    base_accuracy_on_ood_data: float,
                    avg_orig: bool):
    """
    Fine-tune a neural network using out-of-distribution (OOD) data.

    Parameters:
    net (torch.nn.Module): The neural network model to be fine-tuned.
    train_loader_ood (torch.utils.data.DataLoader): DataLoader for fine-tuning using OOD data.
    test_loader_ood (torch.utils.data.DataLoader): DataLoader for testing using OOD data.
    test_loader (torch.utils.data.DataLoader): DataLoader for regular testing data.
    base_accuracy_on_regular_data (float): Base accuracy on regular testing data before fine-tuning.
    base_accuracy_on_ood_data (float): Base accuracy on OOD testing data before fine-tuning.
    avg_orig (bool): Whether to average the original pretrained model. If True, net is averaged with orig_pretrained_net.

    Returns:
    None
    """
    print(f'Fine-Tune Train')
    accuracy_list_on_original, accuracy_list_on_ood = [base_accuracy_on_regular_data], [base_accuracy_on_ood_data]
    orig_pretrained_net = copy(net) if avg_orig else None
    for _ in tqdm(range(len(train_loader_ood))):
        train(net=net, train_loader=train_loader_ood, epochs=1, iterations=1)

        if avg_orig:
            average_models(net, orig_pretrained_net)
        _, acc = test(net=net, test_loader=test_loader)

        accuracy_list_on_original.append(acc)
        _, acc = test(net=net, test_loader=test_loader_ood)

        accuracy_list_on_ood.append(acc)
        save_iteration_results(accuracy_list_on_ood, accuracy_list_on_original)


def freeze_all_layers_but_last(model, num_layers_to_freeze=1):
    """
    Freeze all layers in the model except the last specified number of layers.

    Parameters:
    model: The neural network model.
    num_layers_to_freeze (int, optional): Number of layers to keep unfrozen from the end. Defaults to 1.

    Returns:
    None
    """
    classification_head_start_index = len(list(model.children())) - num_layers_to_freeze

    # Freeze all layers except the last specified number of layers
    for index, child in enumerate(model.children()):
        if index < classification_head_start_index:
            for param in child.parameters():
                param.requires_grad = False


@torch.no_grad()
def average_models(net: torch.nn.Module, orig_pretrained_net: torch.nn.Module):
    """
    Average the weights of the current network with the original pretrained network.

    Parameters:
    net (torch.nn.Module): The neural network model to be averaged.
    orig_pretrained_net (torch.nn.Module): The original pretrained neural network.

    Returns:
    None
    """
    for p, op in zip(net.parameters(), orig_pretrained_net.parameters()):
        p.data += op.data
        p.data /= 2.0


def save_iteration_results(accuracy_list_on_ood_data: List[float], accuracy_list_on_standard_data: List[float]):
    """
    Save the iteration results of accuracy on OOD and standard data to files.

    Parameters:
    accuracy_list_on_ood_data (List[float]): List of accuracies on OOD data.
    accuracy_list_on_standard_data (List[float]): List of accuracies on standard data.

    Returns:
    None
    """
    with open('accs_on_aug.npy', 'wb') as f:
        np.save(f, np.array(accuracy_list_on_ood_data))

    with open('accs_on_original.npy', 'wb') as f:
        np.save(f, np.array(accuracy_list_on_standard_data))
