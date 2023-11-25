################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-14
################################################################################

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models
import wandb  # TODO: REMOVE ME!!!
from tqdm import tqdm  # TODO: REMOVE ME!!!
from cifar100_utils import get_train_validation_set, get_test_set

def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """
    #######################
    # PUT YOUR CODE HERE  # 
    #######################

    # Get the pretrained ResNet18 model on ImageNet from torchvision.models
    model = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")

    # Randomly initialize and modify the model's last layer for CIFAR100.
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Initializing the weights and biases of the last layer.
    nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)
    nn.init.zeros_(model.fc.bias)

    # Freezing all other layers
    for param in model.parameters():
        param.requires_grad = False
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name=None, debug=False):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
        debug: Whether to use a smaller dataset for debugging or not.
    Returns:
        model: Model that has performed best on the validation set.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Load the datasets
    train_set, val_set = get_train_validation_set(data_dir, augmentation_name=augmentation_name)

    # Use a smaller dataset for debugging
    if debug:
        train_set = data.Subset(train_set, range(10))
        val_set = data.Subset(val_set, range(10))

    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Move the model to the device
    model.to(device)

    # Initialize the optimizer (Adam) to train the last layer of the model.
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)

    # Initialize the loss function (CrossEntropyLoss)
    loss_function = nn.CrossEntropyLoss()

    # Initialize the best accuracy to zero
    best_val_accuracy = 0

    # WandB – Watching gradients and weights
    wandb.watch(model, log="all", log_freq=100)

    # Training loop with validation after each epoch. Save the best model.
    print("Training & eval ...")
    for epoch in tqdm(range(epochs)):
    # for epoch in range(tqdm(epochs)):
        # Set model to training mode
        model.train()
        loss = 0

        # Loop over the training set and train the model.
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = loss_function(outputs, targets)

            # Backward pass
            loss.backward()

            # Update the parameters
            optimizer.step()

        # Loop over the validation set and compute the accuracy.
        val_accuracy = evaluate_model(model, val_loader, device)

        # Save the model if it is the best one on validation.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), checkpoint_name)

        # Log the epoch loss, accuracy and best accuracy to WandB.
        wandb.log({
            "loss": loss,
            "val_accuracy": val_accuracy,
            "best_val_accuracy": best_val_accuracy
        })

    # Load the best model on val accuracy and return it.
    model.load_state_dict(torch.load(checkpoint_name))
    model.to(device)

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set model to evaluation mode (Remember to set it back to training mode in the training loop)
    model.eval()

    # Loop over the dataset and compute the accuracy. Return the accuracy
    # Remember to use torch.no_grad().
    accuracy = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute the accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            accuracy += (predicted == targets).sum().item()
    
    accuracy /= total

    # Log the test accuracy to WandB
    wandb.log({"test_accuracy": accuracy})

    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def main(lr, batch_size, epochs, data_dir, seed, augmentation_name, test_noise, debug):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
        test_noise: Whether to test the model on noisy images or not.
        debug: Whether to use a smaller dataset for debugging or not.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set the seed for reproducibility
    set_seed(seed)

    # Set the device to use for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)

    # Load the model
    model = get_model()

    # Get the augmentation to use

    # WandB – Initialize a new run
    wandb.init(project="DL1 Practical 2", config={
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "augmentation_name": augmentation_name,
        "test_noise": test_noise,
        "debug": debug
    })

    # Train the model
    model = train_model(model, lr, batch_size, epochs, data_dir, f"best_model_{augmentation_name}.pt", device, augmentation_name, debug)

    # Evaluate the model on the test set
    print("Testing ...")
    test_set = get_test_set(data_dir, test_noise)
    if debug:
        test_set = data.Subset(test_set, range(10))
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    test_accuracy = evaluate_model(model, test_loader, device, )
    print("Test accuracy: {0:.2f}".format(test_accuracy * 100))

    # End WandB logging
    wandb.finish()
    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=123, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR100 dataset.')
    parser.add_argument('--augmentation_name', default=None, type=str,
                        help='Augmentation to use.')
    parser.add_argument('--test_noise', default=False, action="store_true",
                        help='Whether to test the model on noisy images or not.')
    # Added own argument to use smaller dataset for debugging
    parser.add_argument('--debug', default=False, action="store_true",
                        help='Whether to use a smaller dataset for debugging or not.')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
