""" Utilities for data processing.
    Author: Serena Chen
"""

import numpy as np
import os, torch
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from MDAnalysis.analysis.dihedrals import Ramachandran
from Bio import SeqIO
from transformers import EsmTokenizer, EsmForMaskedLM

# Lookup tables
aa_3to1 = {
    'MET':'M',
    'ARG':'R',
    'HIS':'H',
    'HSD':'H',
    'LYS':'K',
    'ASP':'D',
    'GLU':'E',
    'SER':'S',
    'THR':'T',
    'ASN':'N',
    'GLN':'Q',
    'CYS':'C',
    'SEC':'U', #Selenocysteine
    'GLY':'G',
    'PRO':'P',
    'ALA':'A',
    'VAL':'V',
    'ILE':'I',
    'PHE':'F',
    'TYR':'Y',
    'TRP':'W',
    'LEU':'L',
    'PYL':'O', # Pyrrolysine
    'XAA':'X', # Unknown
    'GLX':'Z', # Glutamic acid or Glutamine
    'ASX':'B', # Asparagine or Aspartic acid
    'XLE':'J'  # Leucine or Isoleucine
}

aa_1to3   = {value:key for key, value in aa_3to1.items()}

aa_to_int = {
    'M':1,
    'R':2,
    'H':3,
    'K':4,
    'D':5,
    'E':6,
    'S':7,
    'T':8,
    'N':9,
    'Q':10,
    'C':11,
    'U':12, #Selenocysteine
    'G':13,
    'P':14,
    'A':15,
    'V':16,
    'I':17,
    'F':18,
    'Y':19,
    'W':20,
    'L':21,
    'O':22, # Pyrrolysine
    'X':23, # Unknown
    'Z':23, # Glutamic acid or Glutamine
    'B':23, # Asparagine or Aspartic acid
    'J':23, # Leucine or Isoleucine
    'start':24,
    'stop':25
}

int_to_aa = {value:key for key, value in aa_to_int.items()}

def get_aa_3to1():
    """
    Get the lookup table (for easy import)
    """
    return aa_3to1

def get_aa_1to3():
    """
    Get the lookup table (for easy import)
    """
    return aa_1to3

def get_aa_to_int():
    """
    Get the lookup table (for easy import)
    """
    return aa_to_int

def get_int_to_aa():
    """
    Get the lookup table (for easy import)
    """
    return int_to_aa

def pdb2resnum(pdb):
    """
    Get number of residues in a PDB by counting the CA atom
    """
    u0 = mda.Universe(pdb, pdb)
    atomselect = "protein and name CA" #"protein and backbone"
    atomselect_ca = u0.select_atoms(atomselect)
    num_sel_atoms = len(atomselect_ca)
    return(num_sel_atoms)

def pdb2dist_arr(pdb):
    """
    Convert pdb to a pairwise distance array using the CB atom or the CA atom for GLY
    """
    u0 = mda.Universe(pdb, pdb)
    atomselect = "(protein and name CB) or (resname GLY and name CA)"
    atomselect_ref = u0.select_atoms(atomselect)
    num_sel_atoms = len(atomselect_ref)
    #print("Number of selected atoms = "+str(num_sel_atoms)+"\n")
    #calculate distance
    dist_arr = distances.self_distance_array(atomselect_ref.positions)
    return (num_sel_atoms, dist_arr)

def dist_arr2matrix(num_sel_atoms, dist_arr):
    """
    Convert a pairwise distance array to a distance matrix
    """
    dist_mat = np.zeros((num_sel_atoms,num_sel_atoms))
    triu = np.triu_indices_from(dist_mat, k=1)
    dist_mat[triu] = dist_arr
    dist_mat.T[triu] = dist_arr
    return (dist_mat)

def pdb2dihedral(pdb):
    """
    Calculate phi and psi angles of a PDB
    """
    u0 = mda.Universe(pdb, pdb)
    atomselect = "protein"
    r = u0.select_atoms(atomselect)
    R = Ramachandran(r).run()
    phi_angles = R.results.angles[0,:,0] #first dimension is time step. since it's a PDB file, there is only one time step
    psi_angles = R.results.angles[0,:,1]
    
    return (phi_angles, psi_angles)

def pdb2fasta(pdbfl, fastafl):
    """
    Save the AA sequence from a PDB into a FASTA file
    """
    a = open(pdbfl,'r')
    #seq = {}
    seq = []
    for line in a:
        if  line[0:4] == 'ATOM':
            atomtype = str(line[12:16]).replace(' ','')
            aatype = str(line[17:20])
            chainid = str(line[21:22])
            resid = int(line[22:26])
            aatype_1char = get_aa_3to1()[aatype]
            if atomtype == 'CA':
                seq.append(aatype_1char)          
#                if chainid == 'A': #get only chain A seq
#                    seq.append(aatype_1char)
#                else:
#                    print ("Non chain A: {}".format(chainid))
        elif line[0:6] == 'ENDMDL':
            break
    a.close()
    #save the sequence
    fh_out = open(fastafl, "w+")
    fh_out.write(">" + pdbfl + "\n" + ''.join(seq) + "\n")
    fh_out.close()
    return ()

def fasta2esm2embeddings(fastafl, outfl, model_weights):
    """
    Compute embeddings for every residue in a FASTA file by a pre-trained ESM-2 model
    """
    # Load the model and the tokenizer
    model = EsmForMaskedLM.from_pretrained(model_weights)
    tokenizer = EsmTokenizer.from_pretrained(model_weights)

    if torch.cuda.is_available():
        model.eval().cuda()  # disables dropout for deterministic results
        device = "cuda"
    else:
        model.eval()
        device = "cpu"

    # Load the fasta sequence file
    fasta_sequence = SeqIO.parse(open(fastafl), 'fasta')
    for fasta in fasta_sequence:
        pid, seq = fasta.description, str(fasta.seq)
        #Tokenize the sequence
        sequence_tokenized = tokenizer(seq, return_tensors="pt").input_ids.to(device=device)

        tokens_len = np.shape(sequence_tokenized)[1]

        with torch.no_grad():
            results = model(input_ids = sequence_tokenized, output_hidden_states=True)

        token_representations = results.hidden_states[model.config.num_hidden_layers]

        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        residue_representations = token_representations[0, 1 : tokens_len - 1]

        arr_embeddings = residue_representations.cpu().numpy()
        np.savez_compressed(outfl, seq_len=tokens_len-2, embeddings=arr_embeddings)
    return ()