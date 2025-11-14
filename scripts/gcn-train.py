#!/usr/bin/env python3
"""Train a GCN model on custom graphs
   Author: Serena Chen
"""
import os, sys, time
import numpy as np
import torch
from random import Random
from torch_geometric.loader import DataLoader
from model import GCN
from loss import MCC_Loss

######## Define variables ##########
backbone = True
sequence = True
cutoff = 8
bs = 128 #batch size
epochs = 500 #total number of epochs

if backbone == True and sequence == True:
    node_feat_num = 1282
    graph_type = "cutoff" + str(cutoff) + "_phipsi_esm2embeddings" # input graphs for training
    if cutoff == 4: 
        print ("Training: backbone & sequence, 4Å cutoff")
        hidden_neurons = [112] #number of neurons in each hidden layer, i.e., the dimension of output feature vector in each hidden layer
        dropout_rates = [0.276] #% of the neurons in each hidden layer be randomly dropped out; used only in training, see models.py
        optimizer_name = 'Adam'
        lr = 0.003 #learning rate
    elif cutoff == 6: 
        print ("Training: backbone & sequence, 6Å cutoff")
        hidden_neurons = [118]
        dropout_rates = [0.308]
        optimizer_name = 'Adam'
        lr = 0.010 
    elif cutoff == 8: 
        print ("Training: backbone & sequence, 8Å cutoff")
        hidden_neurons = [297]
        dropout_rates = [0.439]
        optimizer_name = 'Adam'
        lr = 0.010
    elif cutoff == 10: 
        print ("Training: backbone & sequence, 10Å cutoff")
        hidden_neurons = [85]
        dropout_rates = [0.648]
        optimizer_name = 'Adam'
        lr = 0.014 
    elif cutoff == 12:
        print ("Training: backbone & sequence, 12Å cutoff")
        hidden_neurons = [322] 
        dropout_rates = [0.295]
        optimizer_name = 'Adam'
        lr = 0.012       
    elif cutoff == 16:
        print ("Training: backbone & sequence, 16Å cutoff")
        hidden_neurons = [59] 
        dropout_rates = [0.263]
        optimizer_name = 'Adam'
        lr = 0.027
    elif cutoff == 20:
        print ("Training: backbone & sequence, 20Å cutoff")
        hidden_neurons = [44]
        dropout_rates = [0.693]
        optimizer_name = 'Adam'
        lr = 0.014
    elif cutoff == 24:
        print ("Training: backbone & sequence, 24Å cutoff")
        hidden_neurons = [9] 
        dropout_rates = [0.507] 
        optimizer_name = 'Adam'
        lr = 0.030
    elif cutoff == "_anno24_else8":
        print ("Training: backbone & sequence, 24Å cutoff between binding and 8Å cutoff else")
        hidden_neurons = [98] 
        dropout_rates = [0.357]
        optimizer_name = 'Adam'
        lr = 0.019
    elif cutoff == "_anno8_else24":
        print ("Training: backbone & sequence, 8Å cutoff between binding and 24Å cutoff else")
        hidden_neurons = [12] 
        dropout_rates = [0.260]
        optimizer_name = 'Adam'
        lr = 0.027

elif backbone == False and sequence == True:
    node_feat_num = 1280
    if cutoff == 8:
        graph_type = "cutoff8_esm2embeddings"
        print ("Training: sequence, 8Å cutoff")
        #hidden_neurons = [297]
        #dropout_rates = [0.439]
        #optimizer_name = 'Adam'
        #lr = 0.010 #learning rate
        
        #HPO
        hidden_neurons = [113]
        dropout_rates = [0.618]
        optimizer_name = 'Adam'
        lr = 0.017
    elif cutoff == "_anno24_else8":
        graph_type = "cutoff" + str(cutoff) + "_esm2embeddings"
        print ("Training: sequence, 24Å cutoff between binding and 8Å cutoff else")
        hidden_neurons = [98] 
        dropout_rates = [0.357]
        optimizer_name = 'Adam'
        lr = 0.019
    elif cutoff == "_pred_anno24_else8":
        graph_type = "cutoff" + str(cutoff) + "_esm2embeddings"
        print ("Training: sequence, 24Å cutoff between predicted binding and 8Å cutoff else")
        hidden_neurons = [105] 
        dropout_rates = [0.489]
        optimizer_name = 'Adam'
        lr = 0.023
elif backbone == True and sequence == False:
    node_feat_num = 2
    if cutoff == 8:
        graph_type = "cutoff8_phipsi"
        print ("Training: backbone, 8Å cutoff")
        #hidden_neurons = [297]
        #dropout_rates = [0.439]
        #optimizer_name = 'Adam'
        #lr = 0.010 #learning rate
        
        #HPO
        hidden_neurons = [124, 113, 8, 102]
        dropout_rates = [0.234, 0.349, 0.406, 0.356]
        optimizer_name = 'RMSprop'
        lr = 0.001
        
dataset_graphs = "../datasets/UP000005640_9606_HUMAN_v4_binding_cutoff8_phipsi_esm2embeddings.pt"

#output
out_models_dir = "../gnns"
out_loss = out_models_dir + "/train_loss.txt"
#####################################

torch.manual_seed(12345)

if not os.path.exists(out_models_dir):
    os.makedirs(out_models_dir)

#load the dataset
train_set = torch.load(dataset_graphs)
    
#shuffle the dataset
Random(42).shuffle(train_set)

#train
torch.autograd.set_detect_anomaly(True)

start_time = time.time()
    
inp_node_features = train_set[0].x.size()[1]
out_labels = train_set[0].y.size()[1]
neurons = [inp_node_features] + hidden_neurons + [out_labels] #neurons in each layer

#define the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(neurons).to(device)

#optimizer
optimizer = getattr(torch.optim, optimizer_name) (model.parameters(), lr=lr)
    
#loss function
loss_fn = MCC_Loss()

# create batches; re-shuffle train set at every epoch to reduce model overfitting
train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)

#training loop
fh_out = open(out_loss, "w+")
for epoch in range(epochs):
    model.train()
        
    running_loss = 0.0
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
            
        running_loss += loss.item()
        
    avg_loss = running_loss / (i + 1)

    #print and save loss
    print("Epoch {0}: train_loss {1:.5f}".format(epoch, avg_loss))
    fh_out.write("Epoch {0}: train_loss {1:.5f}\n".format(epoch, avg_loss))    

    #save model
    model_path = out_models_dir + '/model_{}'.format(epoch)
    torch.save(model.state_dict(), model_path)

elapsed_time = time.time() - start_time

#print and save execution time
print ("Execution time: {}".format( time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
fh_out.write("Execution time: {}".format( time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
    
fh_out.close()
