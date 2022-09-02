import re
import selfies as sf
from rdkit import Chem
from cairosvg import svg2png
from conversion import un_tokenize
from rdkit.Chem import rdDepictor
from IPython.display import SVG, display, HTML
from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D


# Converts SMILES string into an svg for visualization
def smilesToSvg(chem, size):
    mol = Chem.Mol(Chem.MolFromSmiles(chem).ToBinary())

    Chem.Kekulize(mol)
    if not mol.GetNumConformers():
        rdDepictor.Compute2DCoords(mol)

    if size:
        drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])

    else:
        sz = len(chem) * 10 + 30
        drawer = rdMolDraw2D.MolDraw2DSVG(min(400, sz), 200)

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg.replace('svg:', '')


# Takes SMILES string and saves  a visualization
# to the specidfied path
# if size not specified, it will be estimated
def smilesToPng(chem, path, size=None):
    assert path.endswith(".png")

    if size:
        assert type(size) is tuple and len(size) is 2

    chem = un_tokenize(chem)

    svg = smilesToSvg(chem, size)

    svg2png(bytestring=svg, write_to=path)


# Draws a SMILES molecule or reaction
# Separates molecules with dotted lines
# Separates reactants, reagents and products with solid line
# Notably the image is in html, so cannot be copied
# Use smilesToPng or screen snipper to save
def drawSmiles(line, size=None):

    if size:
        assert type(size) is tuple and len(size) is 2

    html = '<div style="white-space: nowrap; display: flex;">'
    parts = line.count(">")

    for i, chems in enumerate(un_tokenize(line).split(">")):

        compound_length = chems.count(".")

        for j, chem in enumerate(chems.split(".")):
            html += smilesToSvg(chem, size)

            if j < compound_length:
                html += '<div style="border-left: 1px dotted;"></div>'

        if i < parts:
            html += '<div style="border-left: 2px solid;"></div>'

    display(HTML(html + '</div>'))
