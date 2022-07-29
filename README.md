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
