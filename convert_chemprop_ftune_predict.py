import argparse
import subprocess
import warnings
import os
warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser(description='chemprop train convert arguments')


parser.add_argument(
    '--trainisc',
    type=str,
    default="input/train.csv",
    metavar='TRAINISC',
    help='the name of the train id smiles and compound file (default: input/train.csv)')

parser.add_argument(
    '--name',
    type=str,
    default="new_family",
    metavar='NAME',
    help='the name of the your protein or protein family (default: new_family)')

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
    smilesClassDict = {}
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
            smilesClassDict[id_smiles_class[1]] = id_smiles_class[2]

    return smilesCompoundDict, compoundSmileDict, smilesClassDict

def read_smiles_predict(FileName):
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

def write_id_smiles(file_path, compoundSmilesDict):

    wf = open(file_path, "w", encoding='utf-8')
    wf.write("compound_id,smiles\n")
    for key, value in compoundSmilesDict.items():
        wf.write(key + "," + value + "\n")
    wf.close()

def write_smiles_class(file_path, smilesClassDict):

    wf = open(file_path, "w", encoding='utf-8')
    wf.write("smiles,class\n")
    for key, value in smilesClassDict.items():
        wf.write(key + "," + value + "\n")
    wf.close()

def write_smiles(file_path, compoundSmilesDict):

    wf = open(file_path, "w", encoding='utf-8')
    wf.write("smiles\n")
    for key, value in compoundSmilesDict.items():
        wf.write(value + "\n")
    wf.close()

if __name__ == '__main__':
    args = parser.parse_args()

    train_file = args.trainisc
    name = args.name
    source_checkpoint = args.sc


    new_family_path = "{}/{}".format(training_files_path, name)

    if not os.path.exists(new_family_path):
        os.makedirs(new_family_path)

    smilesCompoundDict, compoundSmilesDict, smilesClassDict = read_smiles(train_file)



    id_smiles_file = train_file.split(".")[0] + "_id_smiles.csv"
    write_id_smiles(id_smiles_file, compoundSmilesDict)

    smiles_class_file = train_file.split(".")[0] + "_smiles_class.csv"
    write_smiles_class(smiles_class_file, smilesClassDict)

    if not os.path.exists(project_file_path + "/output/"):
        os.makedirs(project_file_path + "/output/")
    train_chemprop_file = project_file_path + "/output/" + train_file.split(".")[0].split("/")[1] + "_chemprop.csv"
    print("chemprop_fingerprint is running")

    cmdCommand = "chemprop_fingerprint --test_path " + smiles_class_file + " --checkpoint_path chemprop/" + source_checkpoint + "_checkpoints/model.pt " \
                        "--preds_path " + train_chemprop_file  # specify your cmd command
    # print(cmdCommand)
    process = subprocess.Popen(cmdCommand.split())
    output, error = process.communicate()

    #create compound feature vector for the new family
    if not os.path.exists(new_family_path + "/compound_feature_vectors/"):
        os.makedirs(new_family_path + "/compound_feature_vectors/")
    tsv_file_name = new_family_path + "/compound_feature_vectors/chemprop.tsv"

    smiles_lst = get_smiles_chemprop(train_chemprop_file)
    compound_lst = get_compound_ids(train_file, smiles_lst)

    read_csv_convert_to_tsv(train_chemprop_file, tsv_file_name, smilesCompoundDict, compound_lst)
    print("Training chemprop file is converted")

    #create comp_targ_binary file for the new family
    write_comp_target_features_combined_binary(new_family_path, compoundSmilesDict, smilesClassDict)
    print("Compound target binary file is created")

    #create train fold file for the new family
    fold_list = create_folds(len(compound_lst))

    if not os.path.exists(new_family_path + "/data/folds/"):
        os.makedirs(new_family_path + "/data/folds/")
    write_folds(new_family_path + "/data/folds/train_fold_setting1.txt", fold_list)
    print("Training fold is created")

    ##############################################PREDICT#########################################

    test_file = args.testisf

    smilesCompoundDict, compoundSmilesDict = read_smiles_predict(test_file)
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