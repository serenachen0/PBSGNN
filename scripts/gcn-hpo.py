#!/usr/bin/env python3
"""Hyperparameter tuning of a GCN model on custom graphs
   Author: Serena Chen
"""
import os, sys
import numpy as np
import torch
import optuna
from optuna.trial import TrialState
import torch.nn as nn
from sklearn.model_selection import KFold
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from model import GCN
from loss import MCC_Loss
from torchmetrics.classification import BinaryMatthewsCorrCoef

######## Define variables ##########
study_name = "cutoff8_phipsi_esm2embeddings_5cv"
#input dataset
dataset_graphs = "../datasets/UP000005640_9606_HUMAN_v4_binding_cutoff8_phipsi_esm2embeddings.pt"

#train
n_splits = 5 # x fold cross validation
bs = 128 #batch size
epochs = 100 #total number of epochs
#####################################

torch.manual_seed(12345)
#load the dataset
dataset = torch.load(dataset_graphs)

#define the model
def define_model(trial, train_set):
    # define neurons/channels
    inp_node_features = train_set[0].x.size()[1]
    out_labels = train_set[0].y.size()[1]
    # We optimize the number of layers, hidden units, and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 8)
    hidden_neurons = []
    dropout_rates = []
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        hidden_neurons.append(out_features)
        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.8)
        dropout_rates.append(p)
    neurons = [inp_node_features] + hidden_neurons + [out_labels] #neurons in each layer
    model = GCN(neurons)
    return (model, dropout_rates)

def objective(trial):
    fold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for fold_idx, (train_idx, valid_idx) in enumerate(fold.split(range(len(dataset)))):
        train_set = torch.utils.data.Subset(dataset, train_idx)
        val_set = torch.utils.data.Subset(dataset, valid_idx)
        # create batches; re-shuffle train set at every epoch to reduce model overfitting
        train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=bs, shuffle=True)

        # Generate the model.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, dropout_rates = define_model(trial, train_set)
        model = model.to(device)

        # Generate the optimizers.
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

        loss_fn = MCC_Loss()
        metric = BinaryMatthewsCorrCoef()

        # Train the model.
        for epoch in range(epochs):
            model.train()

            for i, data in enumerate(train_loader):

                # Zero gradients for every batch
                optimizer.zero_grad()

                # Make predictions for this batch
                out = model(data, dropout_rates)

                # Compute the loss and its gradients
                target = data.y.type(torch.float) #convert target type from int to float
                loss = loss_fn(out, target)
                loss.backward()
                # Adjust learning weights
                optimizer.step()

            # Validation of the model.
            model.eval()
            metric_sum = torch.zeros(train_set[0].y.size()[1])
            with torch.no_grad():
                for j, vdata in enumerate(val_loader):
                    target = vdata.y
                    pred = model(vdata, dropout_rates) #dropouts aren't applied in validation, see model.py
                    metric_sum = metric_sum.add(metric(pred.t(), target.t()))
            score = metric_sum / len(val_loader)
            
        scores.append(score)
        evaluation = np.mean(scores)
        
        trial.report(evaluation, fold_idx)    
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return evaluation

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", study_name=study_name)
    study.optimize(objective, n_trials=100)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
