import re
import deepsmiles
import selfies as sf
from rdkit import Chem

RMV_SPACE = re.compile(r'\s+')


# Removes whitespace and new lines
def un_tokenize(line):
    return re.sub(RMV_SPACE, "", line)


# Convert SMILES representation to InChI representation
# Excessive recursion, broken into 4 cases:
# Text, compounds, reactions, molecules
# Simplistic function that will fail if a single molecule is invalid
def smilesConversion(smiles, convert=lambda x : x):

    # list of molecules/reactions/compounds
    if type(smiles) is list:
        return [smilesConversion(smile, convert) for smile in smiles]

    elif "\n" in smiles:
        # txt file
        new_file = ""

        for smile in smiles.split("\n"):
            new_file = new_file + smilesConversion(smile, convert) + "\n"

        return new_file[:-1]

    elif "." in smiles:
        # String containing multiple SMILES molecules
        compound = ""

        for smile in smiles.split("."):
            compound = compound + smilesConversion(smile, convert) + " . "

        return compound[:-3]

    elif ">" in smiles:
        # String broken into reactants and products (and reagents)
        reaction = ""

        for smile in smiles.split("."):
            reaction = reaction + smilesConversion(smile, convert) + " . "

        return reaction[:-3]

    else:
        # Assume string
        smiles = un_tokenize(smiles)
        return convert(smiles)


# InchI conversion
def smilesToInchI(smiles):
    convert = lambda x : Chem.MolToInchi(Chem.MolFromSmiles(x))
    return smilesConversion(smiles, convert)


# SMILES to SELFIES conversion
def smilesToSelfies(smiles):
    return smilesConversion(smiles, sf.encoder)


# SELFIES to SMILES conversion
def selfiesToSmiles(selfies):
    return smilesConversion(selfies, sf.decoder)


# DeepSMILES conversion
def smilesToDeepSmiles(smiles):
    converter = deepsmiles.Converter(rings=True, branches=True)
    return smilesConversion(smiles, converter)
