import re
import os
import deepsmiles
import selfies as sf
from tqdm import tqdm
from collections import Counter
from random import randint, seed, shuffle
from rdkit.Chem import MolFromSmiles, MolToSmiles

# Tokenizes a SELFIES string
def sf_tokenizer(selfies):
    return selfies.replace("][", "] [")


# Taken from Molecular Transformer - https://github.com/pschwllr/MolecularTransformer
# Due to the syntax of DeepSMILES, can be used to transform DEeepSMILES sentences as well
# Though it might be interesting to see how DeepSMILES based models perform when brackets are tokenized together
# Would there still be as many issues with faulty bracket numbers
def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    pattern = '(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.' \
              '|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


# Returns all files shared between specified folders
# At specified path of specified type
def getSharedFile(folders, path="", type_=""):
    global_files = set()

    for i, folder in enumerate(folders):
        local_files = set()

        for file in os.listdir(path + folder):

            if file.endswith(type_):
                local_files.add(file)

        if i == 0:
            global_files = local_files
        else:
            global_files.intersection_update(local_files)

    return global_files

# Returns all files of type 'file_type' from a path
# If a path is a folder, it will search the folder
# If a path is a file of the specified type it will be returned
def filesFromPath(path, file_type):
    assert os.path.exists(path)

    if path.endswith("." + file_type):
        files = [path]
    else:
        assert os.path.isdir(path)
        files = [path + file for file in getSharedFile([path], type_=".txt")]

    return files


# Removes all substrings in differences from text
def rmvDiff(text, differences):

    if differences:

        for difference in differences:
            text = text.replace(difference, "")


# Locates all empty lines in from_file
# and removes all corresponding lines in to_files
# Used to ensure source and target data have a matching pair
# Also ensures that there is no difference in representation datasets
def removeBlanks(from_files, to_files):
    rmv_lines = set()

    for from_file in from_files:

        with open(from_file, 'r') as file:
            lines = file.readlines()
            rmv_lines = rmv_lines.union([i for i, line in enumerate(lines)
                                         if line.isspace()])

    for to_file in to_files:

        with open(to_file, 'r') as file:
            lines = file.readlines()

        with open(to_file, 'w') as file:
            file.writelines([line for i, line in enumerate(lines)
                             if i not in rmv_lines])

    return file
  
