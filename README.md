# SmilesOrSelfies
Predicting chemical reactions with different representations of chemicals structures


## There are 2 methods:
OpenNMT-tf and OpenNMT-py for different PC/GPU capabilities. 
* OpenNMT-py requires PyTorch<=1.6.0, meaning max of CUDA version 10.2. 
  * The documentation will demonstrate how to run on a local machine.
* OpenNMT-tf requires TensorFlow 2.6, 2.7, 2.8, or 2.9, which gives broader CUDA compatability.
  * The documentation will demonstrate how to run on an AWS EC2 instance.
OpenNMT-py will contain an explanation on how to run from your own PC/ 



* [**USPTO_MIT** dataset](https://github.com/wengong-jin/nips17-rexgen) USPTO/data.zip
* [**USPTO_STEREO** dataset](https://ibm.box.com/v/ReactionSeq2SeqDataset) US_patents_1976-Sep2016_*

Both are subsets from data extracted and originally published by Daniel Lowe (many thanks for that!).
preprocess.py was taken from MolecuarTransformer (https://github.com/pschwllr/MolecularTransformer)

Download data from here: https://ibm.box.com/v/MolecularTransformerData

Update your Nvidia Driver and ascertain it's compatability with CUDA versions.
If your PC does not have a GPU or has a GPU incompatible with CUDA<=10.2, follow the instructions for openNMT-tf a 


https://github.com/pschwllr/MolecularTransformer


Step 1) Create Instance CUDA 10.1/10.2 compatable instance (Identify the GPU and check https://www.nvidia.com/Download/Find.aspx). For ease choose an AMI with CUDA 10.2 pre-installed (Deep Learning AMI (Amazon Linux 2) Version 49.0), this means g5's arent compatable as they use

Step 2) launch instance

Ensure correct Cuda version (with above you need to change version, by deleting old and copying 10.2 into folder)

sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-10.2 /usr/local/cuda
export PATH=/home/ubuntu/.local/bin:$PATH



SET UP ANACONDA - this is an oddly complex task - https://phoenixnap.com/kb/how-to-install-anaconda-ubuntu-18-04-or-20-04
Slect right version - hashes https://docs.anaconda.com/anaconda/install/hashes/Anaconda3-2019.03-Linux-x86_64.sh-hash/

export PATH=/home/ubuntu/anaconda3/bin:$PATH
export PATH=/home/ubuntu/anaconda3/bin:$PATH


## Set up environment
```
conda update conda
Create conda -n [NAME] python=3.6


conda activate [NAME]
conda install rdkit -c rdkit
conda install future six tqdm pandas -y
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
conda install pip git -y
pip install selfies
```
### Set up OpenNMT (Current version)
```
git clone https://github.com/OpenNMT/OpenNMT-py.git
```











conda install git
git clone
conda install pytorch line
conda install requirements

preprocess
generate vocab
train
inference
test
nohup & so can be used simultaneously 


opennmt-tf
Cuda requirement

g5 may be most cost efficeint
