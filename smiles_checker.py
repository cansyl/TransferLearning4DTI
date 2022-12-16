from rdkit import Chem
import argparse
import warnings

warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser(description='smiles checker arguments')


parser.add_argument(
    '--sf',
    type=str,
    default="input/test_id_smiles.csv",
    metavar='SF',
    help='the name of smiles file (default: input/test_id_smiles.csv)')

def read_smiles(FileName):
    smilesCompoundDict = {}
    c = 0
    with open(FileName) as f:
        lines = f.readlines()
        for line in lines:
            if c == 0:
                c += 1
                continue
            compound_smile_pair = line.rstrip('\n').split(",")
            smilesCompoundDict[compound_smile_pair[0]] = compound_smile_pair[1]

    return smilesCompoundDict

if __name__ == '__main__':
    args = parser.parse_args()

    smiles_file = args.sf

    smilesCompoundDict = read_smiles(smiles_file)

    invalid_chem_lst, invalid_smiles_lst = [], []
    for key, value in smilesCompoundDict.items():

        smi = value

        m = Chem.MolFromSmiles(smi, sanitize=False)
        if m is None:
            invalid_smiles_lst.append(key)
        else:
            try:
                Chem.SanitizeMol(m)
            except:
                invalid_chem_lst.append(key)

    if len(invalid_chem_lst) > 0:
        print("The following id(s) have invalid chemistry:", invalid_chem_lst)

    if len(invalid_smiles_lst) > 0:
        print("The following id(s) have invalid smiles:", invalid_smiles_lst)