# segNT sample run code
This is a sample code to show how to load and use pretrained Nucleotide Transformers and SegmentNT to generate prediction on input sequence and extract embeddings

It assume you are working on midway3 with gpu. Other running config can be changed in the ***submit.sbatch*** file
## 0. create a new working environment for gpn with mamba
```
mamba create -n segNT python=3.9
mamba activate segNT
```
## 1. clone this repo to your working directory
```
git clone https://github.com/instadeepai/nucleotide-transformer
cd nucleotide-transformer
```
## 2. Install the package needed, also update jax to support cuda
```
pip install .
pip install jax[cuda]
```
## 3. put the files in this repo to this directory

## 4. run the script with sbatch
```
sbatch submit.sbatch
```
The print-out from this file will be stored in output file *log*
