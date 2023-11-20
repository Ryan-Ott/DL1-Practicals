#
# Copyright (c) 2023 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2023
# Date Created: 2023-11-01
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim

# Added imports
import pandas as pd
import matplotlib.pyplot as plt


def plot_curves(train_losses, val_accuracies, test_accuracy):
    plt.figure()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle('PyTorch MLP', fontsize=16)

    # == Training Loss Curve ==
    ax1.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curve')
    ax1.legend()

    # == Validation Accuracy Curve ==
    ax2.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy Curve')

    # == Test Accuracy ==
    ax2.annotate(f"Test Accuracy: {test_accuracy * 100:.2f}%", xy=(1, 0), xycoords='axes fraction', fontsize=12,
                xytext=(-10, 10), textcoords='offset points', ha='right', va='bottom')
    ax2.legend()

    # Set x-axis ticks
    ax1.set_xticks(range(1, len(train_losses) + 1))
    ax2.set_xticks(range(1, len(val_accuracies) + 1))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Saving the plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'PyTorch Curves.png'), dpi=300)
    plt.close(fig)


def plot_confusion_matrix(conf_mat, classes):
    plt.figure(figsize=(10, 10))
    plt.suptitle("PyTorch Confusion Matrix", fontsize=16)
    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            plt.text(j, i, int(conf_mat[i, j]),
                     horizontalalignment="center",
                     color="white" if conf_mat[i, j] > conf_mat.max() / 2. else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Saving the plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'PyTorch Confusion Matrix.png'), dpi=300)
    plt.close()


def plot_metrics_table(confusion_matrix, classes, betas=[0.1, 1, 10]):
    # Calculate metrics
    metrics_default = confusion_matrix_to_metrics(confusion_matrix)
    precision = np.round(metrics_default['precision'], 3)
    recall = np.round(metrics_default['recall'], 3)

    # Create a DataFrame
    data = {
        "Precision": precision,
        "Recall": recall
    }

    for beta in betas:
        f1_beta = np.round(confusion_matrix_to_metrics(confusion_matrix, beta=beta)['f1_beta'], 3)
        data[f"F1-{beta}"] = f1_beta

    df = pd.DataFrame(data, index=classes)

    # == Plotting the table ==
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    ax.set_title("Metrics Table for PyTorch", fontsize=16, weight='bold', pad=15)
    ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center')

    # Saving the plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'PyTorch Metrics Table.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    n_classes = predictions.shape[1]
    conf_mat = np.zeros((n_classes, n_classes), dtype=np.float32)

    for i, probs in enumerate(predictions):
        true = targets[i]
        pred = np.argmax(probs)
        conf_mat[true, pred] += 1
    #######################
    # END OF YOUR CODE    #
    #######################
    return conf_mat


def confusion_matrix_to_metrics(confusion_matrix, beta=1.):
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    TP = np.diag(confusion_matrix)
    FP = np.sum(confusion_matrix, axis=0) - TP
    FN = np.sum(confusion_matrix, axis=1) - TP

    accuracy = np.sum(TP) / np.sum(confusion_matrix)  # accuracy over entire confusion matrix
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_beta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_beta': f1_beta
    }
    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def evaluate_model(model, data_loader, num_classes=10):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    total_conf_mat = np.zeros((num_classes, num_classes), dtype=np.float32)

    with torch.no_grad():
      for x, y in data_loader:
          x = x.to(device)
          y = y.cpu().numpy()
          y_pred = model(x)
          y_pred = y_pred.cpu().numpy()
          total_conf_mat += confusion_matrix(y_pred, y)
      
    metrics = confusion_matrix_to_metrics(total_conf_mat)
    metrics['confusion_matrix'] = total_conf_mat
    # END OF YOUR CODE    #
    #######################
    return metrics


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation.
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    print(f"Using device: {device}")
    # TODO: Initialize model and loss module
    n_inputs = np.prod(cifar10['train'][0][0].shape)  # (C, H, W) (3, 32, 32)
    model = MLP(n_inputs, hidden_dims, n_classes=10, use_batch_norm=use_batch_norm).to(device)
    loss_module = nn.CrossEntropyLoss()
    # TODO: Training loop including validation
    # TODO: Do optimization with the simple SGD optimizer
    val_accuracies = []
    best_val_accuracy = 0.0
    best_model = None
    train_losses = []
    
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # === Training ===
    for epoch in tqdm(range(epochs)):
        sum_loss = 0.0
        num_batches = 0

        for x, y in cifar10_loader['train']:
            # == Forward pass ==
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_module(y_pred, y)
            sum_loss += loss.item()
            num_batches += 1

            # == Backward pass ==
            loss.backward()

            # == Gradient Descent Step ==
            optimizer.step()
        
        avg_train_loss = sum_loss / num_batches
        train_losses.append(avg_train_loss)

        # === Validation ===
        val_metrics = evaluate_model(model, cifar10_loader['validation'])
        val_accuracy = val_metrics['accuracy']
        val_accuracies.append(val_accuracy)

        # === Save best model ===
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = deepcopy(model)

    # TODO: Test best model
    logging_info = evaluate_model(best_model, cifar10_loader['test'])
    # TODO: Add any information you might want to save for plotting
    logging_info['train_losses'] = train_losses
    test_accuracy = logging_info['accuracy']
    model = best_model
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_info


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    best_model, val_accuracies, test_accuracy, logging_info = train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here    
    
    classes = ["airplanes", "cars", "birds", "cats", "deer", "dogs", "frogs", "horses", "ships", "trucks"]
    
    # === Plotting ===
    plot_curves(logging_info['train_losses'], val_accuracies, test_accuracy)

    # === Plotting Confusion Matrix ===
    plot_confusion_matrix(logging_info['confusion_matrix'], classes)
    
    # === Precision, Recall & F1-beta Scores ===
    plot_metrics_table(logging_info['confusion_matrix'], classes)