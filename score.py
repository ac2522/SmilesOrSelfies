import re
import deepsmiles
import numpy as np
import pandas as pd
import selfies as sf
import matplotlib.pyplot as plt
from rdkit.Chem import MolFromSmiles, MolToSmiles


ERROR_TYPE = ["SYNTAX issue:", "Additional product:",
              "Not enough products:", "Wrong product:"]
RMV_SPACE = re.compile(r'\s+')
CONVERTER = deepsmiles.Converter(rings=True, branches=True)


def un_tokenize(line):
    return re.sub(RMV_SPACE, "", line)


def canonDeepSmiles(mol):
    return MolToSmiles(MolFromSmiles(CONVERTER.decode(mol)),
                       isomericSmiles=True)


def canonSelfies(mol):
    return MolToSmiles(MolFromSmiles(sf.decoder(mol)), isomericSmiles=True)


def canonSMILES(mol):
    return MolToSmiles(MolFromSmiles(mol), isomericSmiles=True)


# Prints only if verbose=True
def toPrint(statement, verbose=0):
    if verbose > 0:
        print(statement)


def print_errors(test_data, canon_smiles, verbose=2):
    errors = [0, 0, 0, 0]

    for i, line in enumerate(test_data[0]):

        tgt_line = test_data["tgt"][i]
        tgt = set([canon_smiles(mol) for mol in tgt_line.split(".")])

        try:
            pred = set([canon_smiles(mol) for mol in line.split(".")])

            if tgt.issubset(pred):

                if not pred.issubset(tgt):
                    error = 1
                else:
                    # No error
                    error = -1
            else:
                if len(tgt) > len(pred):
                    error = 2
                else:
                    error = 3

        except Exception as e:
            error = 0

        if error != -1:
            errors[error] += 1

            if verbose == 3:
                print(ERROR_TYPE[error])
                print("tgt:", tgt_line)
                print("pred:", line, "\n")

    print("___ERROR TYPES___")
    print(ERROR_TYPE[0], errors[0])
    print(ERROR_TYPE[1], errors[1])
    print(ERROR_TYPE[2], errors[2])
    print(ERROR_TYPE[3], errors[3], "\n")


# Retrieves accuracy and error percentages for n_best predictions
# test_results is the output of onmt.trasnslator.py
# Targets is the actual set of reaction products
# Where ns is the set of different n_best values to explore
# beam is the number of predictions for each equation
# canon_smiles, is the function required to transform data to canonical SMILES
def inferenceAnalysis(test_results, targets, beam,
                      canon_smiles, ns, verbose=1):
    assert max(ns) <= beam
    assert test_results.endswith(".txt") and targets.endswith(".txt")

    def canonLine(line):
        try:
            return set([canon_smiles(mol) for mol in line.split(".")])

        except Exception as e:
            return {-2}

    with open(test_results, "r") as file:
        test_data = pd.DataFrame(file.readlines(), columns=["string"])

    test_data['idx'] = test_data.index // beam
    test_data = test_data.groupby('idx')['string'].apply(lambda x: pd.Series(x.values)).unstack()
    total = len(test_data) / 100

    with open(targets, "r") as file:
        test_data["tgt"] = file.readlines()

    # Removes spaces and new lines
    test_data = test_data.applymap(un_tokenize)

    if verbose > 1:
        print_errors(test_data[[0, "tgt"]], canon_smiles, verbose)

    # Transforms represntations into canonical SMILES
    test_data = test_data.applymap(canonLine)
    # Checks correctness of products predicted
    results = test_data.apply(lambda row: [row[i].issubset(row["tgt"]) for i in range(10)],
                              axis=1, result_type='expand')

    accuracy = []
    errors = []

    toPrint("Accuracy:", verbose)
    for guesses in ns:
        acc = results[range(guesses)].any(axis=1).sum() / total
        accuracy.append(acc)
        toPrint(f'top {guesses}: {acc}%', verbose)

    toPrint("\nErrors:", verbose)
    for guesses in ns:
        err = sum(test_data[range(guesses)].stack() == {-2}) / total / guesses
        errors.append(err)
        toPrint(f'top {guesses}: {err}%', verbose)

    return accuracy, errors
