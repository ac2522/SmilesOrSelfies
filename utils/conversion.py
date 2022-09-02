import re
import deepsmiles
import selfies as sf
from rdkit import Chem

RMV_SPACE = re.compile(r'\s+')


# Removes whitespace and new line
def un_tokenize(line):
    return re.sub(RMV_SPACE, "", line)


# Convert SMILES representation to InChI representation
# Excessive recursion, broken into 4 cases:
# Text, compounds, reactions, molecules
def smilesConversion(smiles, convert=lambda x : x):

    # list of molecules/reactions/compounds
    if type(smiles) is list:
        return [smilesConversion(smile) for smile in smiles]

    elif "\n" in smiles:
        # txt file
        new_file = ""

        for smile in smiles.split("\n"):
            new_file = new_file + smilesConversion(smile) + "\n"

        return new_file[:-1]

    elif "." in smiles:
        # String containing multiple SMILES molecules
        compound = ""

        for smile in smiles.split("."):
            compound = compound + smilesConversion(smile) + " . "

        return compound[:-3]

    elif ">" in smiles:
        # String broken into reactants and products (and reagents)
        reaction = ""

        for smile in smiles.split("."):
            reaction = reaction + smilesConversion(smile) + " . "

        return inch[:-3]

    else:
        # Assume string
        smiles = un_tokenize(smiles)
        return convert(smiles)


# InchI conversion
def smilesToInchI(smiles):
    convert = lambda x : Chem.MolToInchi(Chem.MolFromSmiles(x))
    return smilesConversion(smiles, convert)


# SELFIES conversion
def smilesToSelfies(smiles):
    return smilesConversion(smiles, sf.encoder)


# DeepSMILES conversion
def smilesToSelfies(smiles):
    converter = deepsmiles.Converter(rings=True, branches=True)
    return smilesConversion(smiles, converter)
