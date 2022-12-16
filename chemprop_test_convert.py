import argparse
import warnings

warnings.filterwarnings(action='ignore')
parser = argparse.ArgumentParser(description='chemprop test convert arguments')

parser.add_argument(
    '--tis',
    type=str,
    default="input/test_id_smiles.csv",
    metavar='TIS',
    help='the name of the test id smiles file (default: input/test_id_smiles.csv)')

parser.add_argument(
    '--tsc',
    type=str,
    default="output/test_smiles_chemprop.csv",
    metavar='TSC',
    help='the name of the test smiles chemprop file (default: output/test_smiles_chemprop.csv)')

def get_compound_ids(FileName):
    compound_lst = []
    with open(FileName) as f:
        lines = f.readlines()
        for line in lines:
            feature_lst = line.rstrip('\n').split(",")
            compound_lst.append(feature_lst[0])
    f.close()
    return compound_lst

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
            smilesCompoundDict[compound_smile_pair[1]] = compound_smile_pair[0]

    return smilesCompoundDict

def read_csv_convert_to_tsv(csv_file_name, tsv_file_name, compoundSmileDict, compound_lst):
    with open(tsv_file_name, 'w') as wf:
        with open(csv_file_name) as f:
            lines = f.readlines()
            for line in lines:
                if "smiles" in line:
                    continue
                else:
                    csv_line = line.rstrip('\n').split(",")
                    comp_id = compoundSmileDict[csv_line[0]]
                    if comp_id in compound_lst:
                        wf.write(comp_id + "\t" + "\t".join([str(float(dim)) for dim in csv_line[1:]]) + "\n")

    f.close()
    wf.close()

if __name__ == '__main__':
    args = parser.parse_args()

    id_smiles_file = args.tis
    test_file = args.tsc

    smilesCompoundDict = read_smiles(id_smiles_file)
    compound_lst = get_compound_ids(id_smiles_file)
    tsv_file_name = test_file.split(".")[0] + ".tsv"
    read_csv_convert_to_tsv(test_file, tsv_file_name, smilesCompoundDict, compound_lst)
