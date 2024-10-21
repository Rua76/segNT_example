# segNT sample run code
This is a sample code to show how to load and use pretrained Nucleotide Transformers and SegmentNT to generate prediction on input sequence and extract embeddings

**Note: the NT transformer and segNT are written with a framework called Jax, which I don't quite familiar. By default they run with CPU. I managed to make NT transformer to run on GPU, but segNT is still on CPU**

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
## 4. Run model downloading script to access pretrained models
As script submitted to the cluster cannot access internet, you need to run the ```download_models.py``` script to download pretrained models before actural task.
```
python download_models.py
```
this script will download one NT transformer model and one Segment NT model to the ```~/.cache``` directory. There is a list of models you can access. 

**Warning: the models are quite big (~5-10GB), downloading too much models might cause *Disk quota* issue. You can always check your quota with command ```quota```. If you ran into such issue, you can delete ```~/.cache/nucleotide_transformer/``` directory to free up space**
## 5. run the script with sbatch
```
sbatch submit.sbatch
```
The print-out from this file will be stored in output file *log*
