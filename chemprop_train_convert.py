import argparse
import warnings
import os
warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser(description='chemprop train convert arguments')


parser.add_argument(
    '--tis',
    type=str,
    default="input/train_id_smiles.csv",
    metavar='TIS',
    help='the name of the train id smiles file (default: input/train_id_smiles.csv)')

parser.add_argument(
    '--tsc',
    type=str,
    default="output/train_smiles_chemprop.csv",
    metavar='TSC',
    help='the name of the train smiles chemprop file (default: output/train_smiles_chemprop.csv)')

parser.add_argument(
    '--scf',
    type=str,
    default="input/train_smiles_class.csv",
    metavar='SCF',
    help='the name of the train similes class file (default: input/train_smiles_class.csv)')

parser.add_argument(
    '--name',
    type=str,
    default="new_family",
    metavar='TF',
    help='the name of the new protein or protein family (default: new_family)')

cwd = os.getcwd()
project_file_path = "{}TransferLearning4DTI".format(cwd.split("TransferLearning4DTI")[0])
training_files_path = "{}TransferLearning4DTI/training_files".format(cwd.split("TransferLearning4DTI")[0])

def get_compound_ids(FileName):
    compound_lst = []
    with open(FileName) as f:
        lines = f.readlines()
        for line in lines:
            feature_lst = line.rstrip('\n').split(",")
            if "compound" in line:
                continue
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

def read_smiles_v2(FileName):
    compoundSmileDict = {}
    c = 0
    with open(FileName) as f:
        lines = f.readlines()
        for line in lines:
            if c == 0:
                c += 1
                continue
            compound_smile_pair = line.rstrip('\n').split(",")
            compoundSmileDict[compound_smile_pair[0]] = compound_smile_pair[1]

    return compoundSmileDict

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

def write_comp_target_features_combined_binary(new_family_path, compoundSmilesDict, smilesClassDict):

    wf = open(new_family_path + "/comp_targ_binary.tsv", "w", encoding='utf-8')

    for key, value in compoundSmilesDict.items():
        if smilesClassDict[value] == "1":
            wf.write("dummy_protein" + "\t" + key + "\t1\n")
        elif smilesClassDict[value] == "0":
            wf.write("dummy_protein" + "\t" + key + "\t0\n")

    wf.close()


def create_folds(length):
    import random
    dtiList = []
    for i in range(0, length):
        dtiList.append(i)
    random.seed(69)
    random.shuffle(dtiList)
    return dtiList


def write_folds(FileName, dti_list):
    f = open(FileName, "w")
    f.write(str(dti_list))
    f.close()

if __name__ == '__main__':
    args = parser.parse_args()

    train_id_smiles_file = args.tis
    train_smiles_chemprop_file = args.tsc
    train_smiles_class_file = args.scf

    name = args.name

    new_family_path = "{}/{}".format(training_files_path, name)

    if not os.path.exists(new_family_path):
        os.makedirs(new_family_path)

    smilesCompoundDict = read_smiles(train_id_smiles_file)
    compound_lst = get_compound_ids(train_id_smiles_file)
    smilesClassDict = read_smiles_v2(train_smiles_class_file)
    compoundSmilesDict = read_smiles_v2(train_id_smiles_file)

    #create comp_targ_binary file for the new family
    write_comp_target_features_combined_binary(new_family_path, compoundSmilesDict, smilesClassDict)
    print("Compound target binary file is created")

    #create train fold file for the new family
    fold_list = create_folds(len(compound_lst))

    if not os.path.exists(new_family_path + "/data/folds/"):
        os.makedirs(new_family_path + "/data/folds/")
    write_folds(new_family_path + "/data/folds/train_fold_setting1.txt", fold_list)
    print("Training fold is created")

    #create compound feature vector for the new family
    if not os.path.exists(new_family_path + "/compound_feature_vectors/"):
        os.makedirs(new_family_path + "/compound_feature_vectors/")
    tsv_file_name = new_family_path + "/compound_feature_vectors/" + "chemprop.tsv"

    read_csv_convert_to_tsv(train_smiles_chemprop_file, tsv_file_name, smilesCompoundDict, compound_lst)
    print("Training chemprop file is converted")
