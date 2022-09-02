# SmilesOrSelfies
Predicting chemical reactions with different representations of chemicals structures.

The paramaterization of this model has been taken from https://github.com/pschwllr/MolecularTransformer.

## DATA
* [**USPTO_MIT** dataset](https://github.com/wengong-jin/nips17-rexgen)
* [**USPTO_STEREO** dataset](https://ibm.box.com/v/ReactionSeq2SeqDataset) 
* [*Handmade** dataset](https://github.com/ac2522/SmilesOrSelfies/data)

Originally the model was trained on a combination of MIT and USPTO, but since they are both extracted from the same database, the code was changed to only train on STEREO. Both USPTO datasets are filtered and modified from US reaction patent data extracted by Daniel Lowe. The main difference being that the MIT dataset filters out stereochemistry. The data can be downloaded from here: https://ibm.box.com/v/MolecularTransformerData

The handmade data, is made up of basic equations outside of the training data chemical space. Most of the reactions do not contain organic compounds.

## There are 2 methods:
For different PC / GPU capabilities. 
* OpenNMT-tf requires TensorFlow 2.6, 2.7, 2.8, or 2.9, which gives broader CUDA compatability.
  * The documentation will demonstrate how to run on a local machine.
* OpenNMT-py requires PyTorch<=1.6.0, meaning max of CUDA version 10.2. 
  * The documentation will demonstrate how to run on an AWS EC2 instance.


Update your Nvidia Driver and ascertain it's compatability with CUDA versions: https://www.nvidia.com/Download/Find.aspx

Also it's worth condisdering the size of the GPU's VRAM, and whether it is sufficient to train a large model. 

If your PC does not have a GPU or has a GPU incompatible with the specified TensorFlow versions, set up with OpenNMT-py.


### OpenNMT-py
This section covers how to set up an environment on AWS EC2. It can obviously be set up on any server, but AWS is the most global (and Google Cloud holds prejudicial views towards students)  

Step 1) Create an instance
* Having set up an AWS account, you will not be allowed to use GPU servers, you must request a quota increase from your console. This may take several days.
  * Also the quota is region locked. So remember what region you requested gpu instance access for
  * Make sure that the region you are requesting hosts the particular instance you weant to use
  * The quota request isn't termed in instances, but in CPU's. Each g4 instance uses a minimum of 8 CPUs. So requesting a quota increase to 4 will take 2 days and you still won't be allowed to launch a g4 instance
* Must be a CUDA 10.1/10.2 compatable instance (Identify the GPU and check https://www.nvidia.com/Download/Find.aspx)
  * This means g5 instances arent compatable, which would be the most cost effective, as they use an A10G GPU
  * I used a g4dn.xlarge, which ironically is the smallest of the g4dn category. It has 16 Gb which is enough for training
* For ease of setup choose an AMI with CUDA 10.2 pre-installed (e.g. Deep Learning AMI (Amazon Linux 2) Version 49.0)
  * The process of installing a CUDA version from scratch is lengthy and prone to errors
* Ensure that it has at least 100Gb drive, this may seem overboard, but the process of adding more is lengthy and can corrupt your files
  * In case you did as I did and need to expand your space https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/recognize-expanded-volume-linux.html#extend-file-system-nvme
* Other stuff: set up a key and secure access. Remember which region you were using.

Step 2) launch instance
* There are many ways to connect and the AWS console provides a number of examples.
  * For convenience reason I used WinSCP for transferring files and Putty for a terminal
   * Both Putty and WinSCP will require you to change the format of the key, which can be done with PuttyGen (a different software than Putty)
* For the Amazon Linux 2 servers, your primary username is ubuntu
  * This is different for Ubuntu based AWS servers, which usually have primary username set to ec2-user
* The address for your EC2 server is listed in details, under 'Public IPv4 DNS'
* You will be charged for every second you server is running, so stop the instance when not in use.
* Occasionally when booting up an  instance, there will be no availability, just wait patiently, twiddle you thumbs.
  * The longest it took for me was 6 hours. But, usually there are no problems.

Step 3) Set up server
* Many of the AMI's such as Deep Learning V49, have multiple CUDA versions installed.
  * Check cuda version with ```nvcc --version```
  * If the starting version isn't correct, you will need to change version, by deleting old and copying 10.1/10.2 into the CUDA folder
   * ```sudo rm /usr/local/cuda```
   * ```sudo ln -s /usr/local/cuda-10.2 /usr/local/cuda```
* Check cuda is running:
  * ```pip3 install torch==1.6.0 torchvision==0.7.0```
   * Your pytorch install command will depend on your CUDA and OS: https://pytorch.org/get-started/previous-versions/  
  * Open Python terminal with ```ipython```
  * ```import torch```
  * ```torch.cuda.is_available```
   * If True then it's working, if False, then you'll need to do some error testing.
  * Top exit ipython ```exit()```  
* Typical Linux commands to initialize environment: 
  * ```sudo apt-get upgrade```
  * ```sudo apt-get update```
  * ```sudo apt-get install curl```
  * ```export PATH=/home/ubuntu/.local/bin:$PATH```
  * Depending on the os, sometimes 'sudo' is replaced by 'yum'
* Setting up Anaconda is an oddly complex task, you can follow the full guide here - https://phoenixnap.com/kb/how-to-install-anaconda-ubuntu-18-04-or-20-04
  * The right Anaconda hash can be found here: https://docs.anaconda.com/anaconda/install/hashes/Anaconda3-2019.03-Linux-x86_64.sh-hash/
  * The commands I used (My Linux expertise is limited, so I can't tell you what did what)
  * ```cd /tmp```
  * ```curl –O https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh```
   * Theres a number of simple prompts, you need to answer yes to.
  * ```sha256sum Anaconda3-2022.05-Linux-x86_64.sh```
  * ```sudo wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh```
  * ```bash Anaconda3-2022.05-Linux-x86_64.sh```
  * ```source ~/.bashrc```
  * ```bash Anaconda3-2022.05-Linux-x86_64.sh```
  * ```export PATH=/home/ubuntu/anaconda3/bin:$PATH```
  * ```conda update conda```

Step 4) Set up environment for OpenNMT-py
* ```conda create -n transformer python=3.6```
  * The Python version has to be compatible with CUDA 10.2, your version of OpenNMT-py and PyTorch 1.6.0
* ```conda activate transformer```
* ```conda install pip```
* ```conda install git```
* ```git clone https://github.com/OpenNMT/OpenNMT-py.git```
* ```git clone https://github.com/ac2522/SmilesOrSelfies.git```
* ```cd OpenNMT-py```
* ```pip3 install -e```
* ```cd ../SmilesOrSelfies```
* ```pip3 install -e```
* ```conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch```
 * Install the right pytorch, from the same link as Step 3) but for Conda this time  
* Move the data folder from /home/ubuntu/SmilesOrSelfies to /home/ubuntu/
* Move the config folder from /home/ubuntu/SmilesOrSelfies to /home/ubuntu/
* Download the USPTO data and place in the data folder
  * For the sake of convenience (and memory) delete all USPTO directories, bar STEREO_mixed and MIT_mixed
   * Within MIT_mixed you might as well delete all but the test data
* Within the data folder create 4 folders: SMILES, SMILES_aug, DeepSMILES, SELFIES
  * SMILES is the base synatax as implemented by the Molecular Transformer
  * SMILES_aug, is the same as SMILES but without canonicalization and with random ordering of molecules
  * DeepSMILES is a modified syntax of SMILES that is meant to improve neural network syntax
  * SELFIES a whole new syntax, designed to be 100% robust

Step 5) Data Preparation
* Since, jupyter notebook was a pain and kept messing up the conda environment, I used ipython and the terminal
* Rename the files in MIT_mixed to start with "mit-"
  * src-test.txt  ->  mit-src-test.txt
* To ensure we are all in the same directory: ```cd```
* ```ipython```
* Prepare data:

```
from SmilesOrSelfies.preprocess import transformSmilesData, smilesToSelfies, smilesToDeepSmiles, checkFormatting
SMILES_PATH = "data/SMILES/"
SELFIES_PATH = "data/SELFIES/"
SMILES_AUG_PATH = "data/SMILES_aug/"
DEEPSMILES_PATH = "data/DeepSMILES/"
NAMES = ['src-train.txt', 'tgt-train.txt', 'src-val.txt', 'tgt-val.txt', 'src-test.txt', 'tgt-test.txt', 'mit-src-test.txt', 'mit-tgt-test.txt', 'common-src-test.txt', 'common-tgt-test.txt']

merge(["data/STEREO_mixed/"], SMILES_PATH, randomize=True, verbose=False)
merge(["data/MIT_mixed/"], SMILES_PATH, randomize=True, verbose=False)
merge(["data/common/"], SMILES_PATH, randomize=False, verbose=False)

transformSmilesData(SMILES_PATH, canonicalizeSmiles, sort=len)

smilesToSelfies(SMILES_PATH, SELFIES_PATH, True, ["tgt", "src"])

smilesToDeepSmiles(SMILES_PATH, DEEPSMILES_PATH, True, ["tgt", "src"])

transformSmilesData(SMILES_PATH, randomizeSmiles, new_path=SMILES_AUG_PATH, shuffle_pos=True)

# Checks all data transformations/conversions worked
checkFormatting(NAMES, 5, [SMILES_PATH, SMILES_AUG_PATH, SELFIES_PATH, DEEPSMILES_PATH])

exit()
```


Step 6) Run the Machine
* We have 4 sets of data and only 1 gpu. We can't run it all at once. So we either:
  * Wait for it to train each
  * OR we set up 4 servers
   * This does not require doing all the above 5 steps again (THANK GOD!)
* To duplicate a selected EC2 instance:
  * Click 'actions' -> 'Image and Templates' -> Create Image
  * Give it a name, then 'Create Image'
  * Create a new instance, the same as the original
  * EXCEPT instead of the Deep Learning AMI, select 'My AMI' and scroll down to find the one you just created
  * It will take some time to load so be patient
* Create a vocabulary (Example for SMILES, but same process for each)
  * ```nohup python OpenNMT-py/build_vocab.py -config configs/SMILES/vocab_config.yaml -n_sample -1 &```
   * ```n_sample``` is the size of vocabulary, and will have an impact on the speed and memory consumption. Setting it to -1, means that all tokens in the source and target data are added to the vocabulary
   * The parametrization for training and forming a vocabulary is done with .yaml config files
   * To allow processes to continue while our connection to the server is down, we use the command ```nohup```. To stop a process from happening: ```kill {PID number}``` - the PID number can be found by running ```nvidia-smi```  
   * The source and target share the same vocabulary
* Train model - this will take roughly 48 hours
  * ```nohup python OpenNMT-py/train.py -config configs/SMILES/train_config.yaml &```
  * If the model stops at any point, you don't have to restart, instead find the last full checkpoint, then:
   * ```nohup python OpenNMT-py/train.py -config configs/SMILES/train_config.yaml -from_checkpoint data/SMILES/model_step_23000.pt &```
  * Run ```nvidia-smi``` to check GPU and memory usage 
* Average last 20 checkpoints
  * ```OpenNMT-py/tools/average_models.py -models model_step_310000.pt model_step_320000.pt model_step_330000.pt model_step_340000.pt model_step_350000.pt model_step_360000.pt model_step_370000.pt model_step_380000.pt model_step_390000.pt model_step_400000.pt model_step_410000.pt model_step_420000.pt model_step_430000.pt model_step_440000.pt model_step_450000.pt model_step_460000.pt model_step_470000.pt model_step_480000.pt model_step_490000.pt model_step_500000.pt -output data/SMILES/model_avg.pt```
* Translate test results (for STEREO, MIT and common datasets) - I was too lazy to write 4 more .yaml files
  * ```nohup  python OpenNMT-py/translate.py -model data/SMILES/model_avg.pt -src src-test.txt -output test_results.txt -beam_size 10 -n_best 10 -batch_size 30 -max_length 500 -gpu 0 &```
  * ```nohup  python OpenNMT-py/translate.py -model data/SMILES/model_avg.pt -src mit-src-test.txt -output mit-test_results.txt -beam_size 10 -n_best 10 -batch_size 30 -max_length 500 -gpu 0 &```
  * ```nohup  python OpenNMT-py/translate.py -model data/SMILES/model_avg.pt -src common-src-test.txt -output common-test_results.txt -beam_size 10 -n_best 10 -batch_size 30 -max_length 500 -gpu 0 &```
   * Beam size has to be greater than n_best
   * N_best is the number of predictions made for each reaction, can be changed as per use case. n_best should be inversely proportional to the batch_size

Step 7) Score the Results
* Combine all the results, by just copying and pasting predictions into one instance
* ```ipython```
```
from score import canonDeepSmiles, canonSMILES, canonSelfies, inferenceAnalysis
import pandas as pd

canons = [canonDeepSmiles, canonSMILES, canonSMILES, canonSelfies]
langs = ["DeepSMILES", "SMILES", "SMILES_aug", "SELFIES"]
ns = [1, 3, 5, 10]

accuracies = []
errors = []
for dataset in ["", "mit-", "common-", ]:
    for canon, lang in zip(canon, langs):
        acc, err = inferenceAnalysis(f'../data/{lang}/{dataset}test_results.txt', '../data/{lang}/{dataset}tgt-test.txt', 10, canon, ns, verbose=0)
        accuracies.append(acc)
        errors.append(err)
    print(dataset)
    print("Accuracy:")
    print(pd.DataFrame(accuracies, index=langs))
    print("Errors:")
    print(pd.DataFrame(errors, index=langs))
```
* This will print out the accuracy and errors for each representation for each database tested upon.


### OpenNMT-tf
Based on a Windows 10 setup. Sorry, I kind of ran out of time for this part.
Step 1)
* Install Cuda
* Install CuDNN
* Install Anaconda
* Setup conda environment
* ```git clone https://github.com/OpenNMT/OpenNMT-tf.git```
* ```git clone https://github.com/ac2522/SmilesOrSelfies.git```

Step 2) 
* Processing the data is the same as OpenNMT-py
* Create a vocabulary
  * May need slight adjustments to paths
* To train run tf_configs/train.py
  * May need slight adjustments to paths
  * Also running the evaluation will need some parametization 

