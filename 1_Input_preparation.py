#parts of code and material come from to Lin, C., et al., Using neural networks for reducing the dimensions of single-cell RNA-Seq data. Nucleic Acids Res, 2017. 45(17): p. e156.'

import numpy as np
from collections import defaultdict
import argparse

#####################################################################################################################
parser = argparse.ArgumentParser(description='MultiCapsNet')

parser.add_argument('--scRNA_seq_data', type=str, default='data/00_mouse_sc_RNAseq_expre_original.npy', help='scRNA_seq_expression_data')
parser.add_argument('--gene_name', type=str, default='data/00NN_training_PPITF_9437_genes.txt', help='gene name in the expression matrix')
parser.add_argument('--TF_ppi_genes', type=str, default='data/00ppi_tf_merge_cluster.txt', help='relationship between gene and TF/ppi')

args = parser.parse_args()

scRNA_seq_data = args.scRNA_seq_data
gene_name = args.gene_name
TF_ppi_genes = args.TF_ppi_genes

gene_names = []
gene_line = open(gene_name ).readlines()

for line in gene_line:
    splits = line.replace('\n', '').split('\t')
    for gene in splits:
        gene_names.append(gene)


group_lines=open(TF_ppi_genes).readlines()
group_gene_index_dict=defaultdict(lambda:[])
for line in group_lines:
    splits=line.replace('\n','').replace('\r','').split('\t')
    gn=splits[0]
    for sp in splits[1:]:
        group_gene_index_dict[gn].append(gene_names.index(sp))

sorted_group_names=sorted(group_gene_index_dict.keys())

key0 = sorted_group_names[0]
inputs = np.load(scRNA_seq_data)
print(inputs.shape)

data_source_length = []
X = inputs[group_gene_index_dict[key0],:]
data_source_length.append(inputs[group_gene_index_dict[key0],:].shape[0])

for key in sorted_group_names:
    if key == key0:
        continue
    data_source_length.append(inputs[group_gene_index_dict[key], :].shape[0])
    X = np.vstack((X,inputs[group_gene_index_dict[key],:]))

np.save('output/preprocessed_data_length.npy',np.asarray(data_source_length))
print(sum(data_source_length))
print(X.shape)
np.save('output/preprocessed_training_data.npy',X)
