# Relation-aware Language-Graph Transformer for Question Answering (AAAI 2023)
Official pytorch implementation of 'Relation-aware Language-Graph Transformer for Question Answering' (AAAI 2023)


## Setup
- Clone repository 
```
git clone https://github.com/mlvlab/QAT.git
cd QAT
```
- Setup conda environment
```
conda create -n QAT python=3.8
conda activate QAT
```
- Install packages with a setup file
```
bash setup.sh
```
- Download data
We use the question answering datasets (CommonsenseQA, OpenBookQA, and MedQA-USMLE) and their knowledge graphs.
We preprocess the dataset following QA-GNN. 
You can download all the preprocessed data with the link.

## Run Experiments
```
bash run_csqa.sh
```

## Acknowledgement
This repo is built upon the QA-GNN and GSC:
```
https://github.com/michiyasunaga/qagnn
https://github.com/kuan-wang/graph-soft-counter
```
