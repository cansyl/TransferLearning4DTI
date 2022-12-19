import argparse
import subprocess
import warnings
import os
warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser(description='chemprop train convert arguments')

parser.add_argument(
    '--testisf',
    type=str,
    default="input/test.csv",
    metavar='TESTISF',
    help='the name of the test id and smiles file (default: input/test.csv)')

parser.add_argument(
    '--sc',
    type=str,
    default="kinase",
    metavar='SC',
    help='the name of the source checkpoint (default: kinase)')

cwd = os.getcwd()
project_file_path = "{}/TransferLearning4DTI".format(cwd.split("TransferLearning4DTI")[0])
training_files_path = "{}/TransferLearning4DTI/training_files".format(cwd.split("TransferLearning4DTI")[0])

def get_compound_ids(FileName, smiles_lst):
    compound_lst = []
    with open(FileName) as f:
        lines = f.readlines()
        for line in lines:
            feature_lst = line.rstrip('\n').split(",")
            if "compound" in line:
                continue
            if feature_lst[1] in smiles_lst:
                compound_lst.append(feature_lst[0])
    f.close()
    return compound_lst

#get trained smiles from the csv file
def get_smiles_chemprop(FileName):
    compound_lst = []
    with open(FileName) as f:
        lines = f.readlines()
        for line in lines:
            feature_lst = line.rstrip('\n').split(",")
            if "smiles" in line:
                continue

            compound_lst.append(feature_lst[0])
    f.close()
    return compound_lst

def read_smiles(FileName):
    smilesCompoundDict = {}
    compoundSmileDict = {}
    c = 0
    with open(FileName) as f:
        lines = f.readlines()
        for line in lines:
            if c == 0:
                c += 1
                continue
            id_smiles_class = line.rstrip('\n').split(",")
            smilesCompoundDict[id_smiles_class[1]] = id_smiles_class[0]
            compoundSmileDict[id_smiles_class[0]] = id_smiles_class[1]

    return smilesCompoundDict, compoundSmileDict


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


def write_smiles(file_path, compoundSmilesDict):

    wf = open(file_path, "w", encoding='utf-8')
    wf.write("smiles\n")
    for key, value in compoundSmilesDict.items():
        wf.write(value + "\n")
    wf.close()

if __name__ == '__main__':
    args = parser.parse_args()

    source_checkpoint = args.sc
    test_file = args.testisf

    smilesCompoundDict, compoundSmilesDict = read_smiles(test_file)
    smiles_file = test_file.split(".")[0] + "_smiles.csv"
    write_smiles(smiles_file, compoundSmilesDict)

    test_chemprop_file = project_file_path + "/output/" + test_file.split(".")[0].split("/")[-1] + "_chemprop.csv"

    print("chemprop_fingerprint is running")

    cmdCommand = "chemprop_fingerprint --test_path " + smiles_file + " --checkpoint_path chemprop/" + source_checkpoint + "_checkpoints/model.pt " \
                                     "--preds_path " + test_chemprop_file  # specify your cmd command
    # print(cmdCommand)
    process = subprocess.Popen(cmdCommand.split())
    output, error = process.communicate()

    smiles_lst = get_smiles_chemprop(test_chemprop_file)
    compound_lst = get_compound_ids(test_file, smiles_lst)

    tsv_file_name = test_chemprop_file.split(".")[0] + ".tsv"

    read_csv_convert_to_tsv(test_chemprop_file, tsv_file_name, smilesCompoundDict, compound_lst)
    print("Test chemprop file is converted")