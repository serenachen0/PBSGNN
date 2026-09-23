#!/usr/bin/env python3
"""Prepare datasets
   Author: Serena Chen
"""

import os, sys
import torch
from torch_geometric.data import Data
import numpy as np
from data_utils import pdb2dist_arr, dist_arr2matrix, pdb2dihedral, pdb2fasta, fasta2esm2embeddings
from tqdm import tqdm

############## Variable settings ################
ds_dir = "../datasets"
pdb_dir = "../pdbs" #folder that stores AFDB PDB files
model_weights = "../weights" #folder that stores ESM-2 pretrained model weights
version = "v4"
proteomeid_dict = {"HUMAN" : "UP000005640_9606",
                   "MOUSE" : "UP000000589_10090",
                   "RAT" : "UP000002494_10116", 
                   "YEAST" : "UP000002311_559292"
                  }
annotation = "binding"
dist_cutoff = 8 #define edges using a cutoff distance
esm2 = True #True/False, True:include pretrained ESM-2 embeddings in node features
phipsi = True #True/False, True:include phi and psi angles in node features
#################################################

#define output file suffix
if esm2:
    if phipsi:
        suffix = "phipsi_esm2embeddings"
    else:
        suffix = "esm2embeddings"
else:
    if phipsi:
        suffix = "phipsi"
    else:
        sys.exit("No node features!")

for organism, proteomeid in proteomeid_dict.items():

    inpfl = ds_dir + "/" + proteomeid + "_" + organism + "_" + version + "_" + annotation + "_all_uniprot.txt"
    outfl = ds_dir + "/" + proteomeid + "_" + organism + "_" + version + "_" + annotation + "_cutoff" + str(dist_cutoff) + "_" + suffix + ".pt"

    # do not overwrite existing output file
    if os.path.exists(outfl):
        print(f"Skipping {organism}: output file already exists: {outfl}")
        continue
    
    #get total number of lines in the inpfl
    with open(inpfl, 'r') as fl:
        next(fl)
        total_ids = sum(1 for line in fl)

    #generate graphs
    dataset = []
    with open(inpfl, 'r') as fl:
        next(fl)  # skip the header line
        pbar = tqdm(fl, total = total_ids)
        for ln, line in enumerate(pbar):
            line = line.rstrip()
            cols = line.split('\t')
            uniprotid = cols[0]
            seqlen = int(cols[1])
            resids_str = cols[2]
            pdbfl = pdb_dir + "/AF-" + uniprotid + "-F1-model_v4.pdb"
            
            if os.path.exists(pdbfl):
                pbar.set_description("Generating graphs {:6n}/{:6n} UniProtID {:12s}".format(ln+1, total_ids, uniprotid))

                #calculate distance array using the CB atom or the CA atom for GLY
                num_sel_atoms, dist_array = pdb2dist_arr(pdbfl)
                #convert the distant array to distance matrix
                dist_mat_array =  dist_arr2matrix(num_sel_atoms, dist_array)

                #generate edge_index (define an edge using a cutoff distance)
                #exclude self (diagonal)
                np.fill_diagonal(dist_mat_array, dist_cutoff + 1)
                #extract i,j pairs where distance < cutoff
                pairs = np.argwhere(dist_mat_array < dist_cutoff)
                edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()

                #get edge features (edge_attr)
                #define edge features using the pairwise distances
                #edge_index [2, num_edges]
                #edge_attr shape [num_edges, num_edge_features], where num_edge_features = 1
                num_edges = edge_index.size()[1]
                dists = dist_mat_array[dist_mat_array < dist_cutoff].reshape(num_edges, 1)
                edge_attr = torch.tensor(dists, dtype=torch.float)

                #get node features (x)
                #x shape [num_nodes, num_node_features]
                if phipsi:
                    #calculate dihedral angles for every residue except for the first and last residues
                    phi, psi = pdb2dihedral(pdbfl)
                    node_ft = np.zeros((num_sel_atoms, 2))
                    node_ft[1:-1,0] = phi
                    node_ft[1:-1,1] = psi
                    node_ft[0,:] = node_ft[1,:]
                    node_ft[-1,:] = node_ft[-2,:]

                if esm2:
                    #create a fasta sequence file from pdbfl and compute the ESM-2 embeddings
                    fastafl =  pdb_dir + '/' + uniprotid + '.fasta'
                    esm2embeddingsfl = pdb_dir + '/' + uniprotid + '_embeddings.npz'
                    pdb2fasta(pdbfl, fastafl)
                    fasta2esm2embeddings(fastafl, esm2embeddingsfl, model_weights)
                    
                    #load AA embeddings predicted by pretrained ESM-2
                    loaded3 = np.load(esm2embeddingsfl)
                    seq_len = loaded3['seq_len'] #this number == num_sel_atoms 
                    aa_probs = loaded3['embeddings'] #a num_sel_atoms x num_neurons (1280) matrix

                    ##### sanity check ####
                    if seq_len != num_sel_atoms:
                        print ("ERROR! {} seq_len (from embeddings, {}) != num_sel_atoms (from distance array, {})".format(uniprotid, seq_len, num_sel_atoms))
                    #######################
                    
                    if phipsi:
                        x_ft = np.concatenate((node_ft, aa_probs), axis=1)
                        x = torch.tensor(x_ft, dtype=torch.float)
                    else:
                        x = torch.tensor(aa_probs, dtype=torch.float)

                else:
                    if phipsi:
                        x = torch.tensor(node_ft, dtype=torch.float)

                #generate y, node-level targets of shape [num_nodes, *]
                target = np.zeros((num_sel_atoms, 1))
                resids = resids_str.split(',')
                for j in resids:
                    idx = int(j)
                    if idx <= num_sel_atoms: #exclude the binding site residues that are not present in the AlphaFold structure
                        target[idx-1] = 1
    
                y = torch.tensor(target, dtype=torch.int)

                #sanity check 
                #if (int(sum(target)) != len(resids)):
                #    print (uniprotid, int(sum(target)), len(resids), np.argwhere(target == 1), resids)

                #save the graph
                data = Data(x=x,
                            edge_index=edge_index, 
                            edge_attr=edge_attr,
                            y=y)
                data.num_nodes = data.y.shape[0]
                data.uniprotid = uniprotid
                dataset.append(data)
                
                #clean up
                if esm2:
                    os.remove(fastafl)
                    os.remove(esm2embeddingsfl)

    torch.save(dataset, outfl) #save all graphs in a file


