##  MultiCapsNet: an interpretable deep learning classifier integrate data from multiple sources

This repository contains the official Keras implementation of:

**MultiCapsNet: an interpretable deep learning classifier integrate data from multiple sources**



**Requirements**
- Python 3.6
- conda 4.4.10
- keras 2.2.4
- tensorflow 1.11.0
- establish the environment through code "conda create -n sccaps python=3.6 keras=2.2.4 tensorflow=1.11.0"


**1. Input_preparation**
- *To unzip the all rar files in dictionary 'data'
- *Decoupling scRNAseq data according to prior knowledge*
```
#In order to meet the requirement of the modeling training and analysis step, the features(genes) of the inputs are ordered according to the prior knowledge
```
- *Augments:*
```
#'--scRNA_seq_data', type=str, default='data/00_mouse_sc_RNAseq_expre_original.npy', help='scRNA_seq_expression_data'
#'--gene_name', type=str, default='data/00NN_training_PPITF_9437_genes.txt', help='gene name in the expression matrix'
#'--TF_ppi_genes', type=str, default='data/00ppi_tf_merge_cluster.txt', help='relationship between gene and TF/ppi'
```

- *Article example*
```
#- (1). mouse single cell RNA-seq dataset:
python 1_Input_preparation.py

#- (2). You could preprocess your own data
```

- *Output*
```
#output/preprocessed_data_length.npy
#output/preprocessed_training_data.npy
```

**2. Model training**

- *Augments:*
```
#'--inputdata', type=str, default='data/1_variant_call_data.npy', help='address for input data'
#'--inputcelltype', type=str, default='data/1_variant_call_type.npy', help='address for celltype label'
#'--source_division', type=str, default='data/1_variant_call_data_length.npy', help='data source length'
#'--num_classes', type=int, default=3, help='number of class need to specify'
#'--randoms', type=int, default=30, help='random number to split dataset'
#'--dim_capsule', type=int, default=4, help='dimension of the capsule'
#'--activation_function', type=str, default='relu', help='activation function for primary capsule'
#'--batch_size', type=int, default=200, help='training parameters_batch_size'
#'--epochs', type=int, default=200, help='training parameters_epochs'
```

- *Article example*
```
#- (1). Variant call dataset:
python 2_model_training.py

#- (2). mouse single cell RNA-seq dataset:
python 2_model_training.py --inputdata data/2_mouse_sc_RNAseq_expre.npy --inputcelltype data/2_mouse_sc_RNAseq_celltype.npy --num_classes 7 --dim_capsule 8 --epochs 10 --source_division data/2_mouse_sc_RNAseq_data_length.npy --activation_function tanh

#- (3). SCENIC mouse brain single cell RNA-seq dataset:
python 2_model_training.py --inputdata data/3_SCENIC_expre.npy --inputcelltype data/3_SCENIC_celltype.npy --num_classes 7 --dim_capsule 8 --epochs 10 --source_division data/3_SCENIC_data_length.npy --activation_function tanh

#- (4). You could train the model with your own data
```

- *Output*
```
#output/Model_training.weights
```

**3. Model analysis**

- *Augments:*
```
#'--inputdata', type=str, default='data/1_variant_call_data.npy', help='address for input data'
#'--inputcelltype', type=str, default='data/1_variant_call_type.npy', help='address for celltype label'
#'--source_division', type=str, default='data/1_variant_call_data_length.npy', help='data source length'
#'--num_classes', type=int, default=3, help='number of class need to specify'
#'--randoms', type=int, default=30, help='random number to split dataset'
#'--dim_capsule', type=int, default=4, help='dimension of the capsule'
#'--activation_function', type=str, default='relu', help='activation function for primary capsule'
#'--training_weights', type=str, default='weights/1_variant_call.weights', help='training_weights'
```
- *Article example*
```
#- (1). Variant call dataset:
python 3_Model_analysis.py

#- (2). mouse single cell RNA-seq dataset:
python 3_Model_analysis.py --inputdata data/2_mouse_sc_RNAseq_expre.npy --inputcelltype data/2_mouse_sc_RNAseq_celltype.npy --num_classes 7 --dim_capsule 8 --source_division data/2_mouse_sc_RNAseq_data_length.npy --activation_function tanh --training_weights weights/2_mouse_sc_RNAseq.weights

#- (3). SCENIC mouse brain single cell RNA-seq dataset:
python 3_Model_analysis.py --inputdata data/3_SCENIC_expre.npy --inputcelltype data/3_SCENIC_celltype.npy --num_classes 7 --dim_capsule 8 --source_division data/3_SCENIC_data_length.npy --activation_function tanh --training_weights weights/3_SCENIC.weights

#- (4). You could analyze the model with your own data and weights
```

- *Output*
```
#output/heatmaps.png
#output/overall_heatmaps.png
#output/top_rank_primary_capsule.txt
```

**capsule networks implementation**

the capsule parts refer to https://github.com/bojone/Capsule and https://kexue.fm/archives/5112