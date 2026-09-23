# PBSGNN
To investigate cutoff distances and node features for protein binding site residue prediction with graph neural networks.

Files required:
- Download PDB files from AlphaFold Protein Structure Database and save the files in ./pdbs
- Download ESM-2 checkpoint esm2_t33_650M_UR50D from Hugging Face and save the files in ./weights

Four organism datasets are provided in ./datasets, each includes a list of UniProtIDs with at least one UniProtKB annotated binding site residue and the corresponding protein sequence length:
- Human - UP000005640_9606_HUMAN_v4_binding_all_uniprot.txt
- Mouse - UP000000589_10090_MOUSE_v4_binding_all_uniprot.txt
- Rat - UP000002494_10116_RAT_v4_binding_all_uniprot.txt
- Yeast - UP000002311_559292_YEAST_v4_binding_all_uniprot.txt

The model checkpoints from the last ten training epochs (model_490–model_499) for each of the 14 GNNs are provided in ./gnns:
File name                     | Cutoff (Å) | Node feature       | Hyperparameter optimized? (Y/N)
| :---------------------------|:----------:|:------------------:| :------------------------------------------------:
model_cutoff4_1282            |4           |Sequence & structure| Y   
model_cutoff6_1282            |6           |Sequence & structure| Y   
model_cutoff8_1282            |8           |Sequence & structure| Y   
model_cutoff10_1282           |10          |Sequence & structure| Y
model_cutoff12_1282           |12          |Sequence & structure| Y
model_cutoff16_1282           |16          |Sequence & structure| Y
model_cutoff20_1282           |20          |Sequence & structure| Y
model_cutoff24_1282           |24          |Sequence & structure| Y
model_cutoff_anno24_else8_1282|24 \| 8     |Sequence & structure| Y
model_cutoff_anno8_else24_1282| 8 \| 24    |Sequence & structure| Y
model_cutoff8_1280            |8           |Sequence            | Y
model_cutoff8_2               |8           |Structure           | Y
model_cutoff8_1280_nohpo      |8           |Sequence            | N, use the hyperparameters of model_cutoff8_1282
model_cutoff8_2_nohpo         |8           |Structure           | N, use the hyperparameters of model_cutoff8_1282

Scripts for graph representation, hyperparameter optimization, training, and evaluation are provided in ./scripts
