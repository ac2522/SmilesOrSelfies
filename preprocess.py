import re
import os
import deepsmiles
import 
as sf
from tqdm import tqdm
from collections import Counter
from random import randint, seed, shuffle
from rdkit.Chem import MolFromSmiles, MolToSmiles


RMV_SPACE = re.compile(r'\s+')


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
# Also ensures that there is no incongruence between
# datasets of different representations
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


# Combines datasets, originally used to merge MIT and STEREO sets
# Also used for randomizing/shuffling data
def merge(folders, new_path="", old_path="", randomize=False,
          seed_no=randint(1, 1000), verbose=True):

    assert type(old_path) is str and type(folders[0]) is str
    assert type(folders) is list and len(folders) > 0
    combine_txts = getSharedFile(folders, path=old_path, type_=".txt")
    assert len(combine_txts) > 0

    for txt in tqdm(combine_txts):
        data = []

        if verbose:
            print(txt)

        for folder in folders:

            with open(old_path + folder + txt, 'r') as file:
                data.extend(file.readlines())
                data[-1] = data[-1] + "\n"
                
        if randomize:
            seed(seed_no)
            shuffle(data)

        data[-1] = data[-1][:-1]

        with open(new_path + txt, 'w') as file:
            file.writelines(data[:-1])

            if verbose:
                print(f'{txt} combined file length: ', data.count("\n") - 1)


# Randomizes SMILES representation
# This method will randomize the representation
def randomizeSmiles(smiles):
    return MolToSmiles(MolFromSmiles(smiles), doRandom=True)


# Canonicalization of a SMILES representation
# This method will transform all representations of the same molecule
# into a single representation
def canonicalizeSmiles(smiles):
    return MolToSmiles(MolFromSmiles(smiles), isomericSmiles=True)


# Transforms a single SMILES line  
def transformSmilesLine(line, func, sort=None, shuffle_pos=False):
    chems = [smi_tokenizer(func(re.sub(RMV_SPACE, '', chem)))
             for chem in line.split('.')]

    if sort:
        chems.sort(key=sort)

    if shuffle_pos:
        shuffle(chems)

    return ' . '.join(chems) + "\n"


# Applies functions to individual lines
# Used either to canonicalize SMILES, or choose a random representation
# Can shuffle or order (setting sort as the sorting method)
def transformSmilesData(path, func, sort=None, new_path=None, shuffle_pos=False):
    assert type(path) is str
    assert not (sort and shuffle_pos)

    files = filesFromPath(path, "txt")

    for file in tqdm(files):

        with open(file, 'r') as read:
            lines = read.readlines()

        if new_path:
            file = file.replace(path, new_path)

        with open(file, 'w') as write:

            for line in lines:
                write.write(transformSmilesLine(line, func, sort, shuffle_pos))


# Converts files of SMILES representations into SELFIES representations.
# All failed conversions will be left as blank lines,
# reconcile_errors removes blank line from SMILES and SELFIES representations.
# diff_between removes blank lines from respective files (e.g. src and tgt):
# otherwise equations and predictions will be misaligned.
def smilesToSelfies(smiles_path, selfies_path, reconcile_errors=False,
                    diff_between=None, verbose=True):

    smilesConversion(smiles_path, selfies_path, sf.encoder, sf_tokenizer,
                     reconcile_errors, diff_between, verbose)
    
    
# Converts files of SMILES representations into DeepSMILES representations.
# All failed conversions will be left as blank lines,
# reconcile_errors removes blank line from SMILES and SELFIES representations.
# diff_between removes blank lines from respective files (e.g. src and tgt):
# otherwise equations and predictions will be misaligned.
def smilesToDeepSmiles(smiles_path, selfies_path, reconcile_errors=False,
                       diff_between=None, verbose=True):

    converter = deepsmiles.Converter(rings=True, branches=True)
    smilesConversion(smiles_path, selfies_path, converter.encode, smi_tokenizer,
                     reconcile_errors, diff_between, verbose)
                
# Converts files of SMILES representations into other representations.
# All failed conversions will be left as blank lines,
# reconcile_errors removes blank line from SMILES and SELFIES representations.
# diff_between removes blank lines from respective files (e.g. src and tgt):
# otherwise equations and predictions will be misaligned.
# new_path is the path for new datasets, reccomended to be different
# from the smiles_path as otherwise SMILES data will be overwriten
# converter and tokenizer are the respective functions for
# conversion and tokenization
def smilesConversion(smiles_path, new_path, converter, tokenizer,
                     reconcile_errors=False, diff_between=None, verbose=True):

    assert type(smiles_path) is str and type(new_path) is str
    files = filesFromPath(smiles_path, "txt")
    total_errors = Counter()

    for i, file in enumerate(tqdm(files)):
        converted_lines = []
        errors = {}

        with open(file, 'r') as smiles_file:
            lines = smiles_file.readlines()

        for line in lines:
            chems = line.split(".")

            try:
                for j, chem in enumerate(chems):
                    chem = ''.join(chem.replace(" ", "").replace("\n", ""))
                    chems[j] = tokenizer(converter(chem))
                converted_lines.append(" . ".join(chems))

            except Exception as e:

                if errors.get(chem):
                    errors[chem] += 1
                else:
                    errors[chem] = 1
                    
                converted_lines.append("")

        with open(new_path + file.split("/")[-1], 'w') as new_file:
            new_file.write('\n'.join(converted_lines))

        if verbose:
            print(file, f'errors: {sum(errors.values())} | '
                  f'Total lines: {len(converted_lines)}')
            total_errors = total_errors + Counter(errors)

    if verbose:
        print("--Conversion complete--\n")
        print(total_errors.most_common())

    # Remove representational dataset incongruences
    if reconcile_errors:

        if diff_between:
            assert type(diff_between) is list and type(diff_between[0])

        for file in tqdm(set([rmvDiff(file, diff_between) for file in files])):
            original_files = [f for f in files if rmvDiff(f, diff_between) == file]
            new_files = [new_path + f.split("/")[-1]
                             for f in rec_files]

            removeBlanks(new_files, original_files + new_files)

        if verbose:
            print("--Errors reconciled--")


# Ensures no formatting errors (or at least the most common ones)
# Partitions indicates the number of data divisions
# (e.g. training, validation and test data)
def checkFormatting(names, partitions, paths=[""]):
    lengths = set()

    for path in paths:
        for name in names:

            with open(path + name, 'r') as file:
                lengths.add(len(file.readlines()))
                assert file.read().count("\n\n") == 0

    assert(len(lengths) < partitions)
