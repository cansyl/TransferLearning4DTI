import os
import random
import argparse
import json
from builtins import len

parser = argparse.ArgumentParser(description='create training and test datasets arguments')

parser.add_argument(
    '--d',
    type=str,
    default="transporter",
    metavar='D',
    help='the name of the dataset (default: transporter)')

parser.add_argument(
    '--ss',
    type=int,
    default=10,
    metavar='SS',
    help='subset size (default: 10)')

cwd = os.getcwd()
training_files_path = "{}TransferLearning4DTI/training_files".format(cwd.split("TransferLearning4DTI")[0])


def read_file_return_total_line(filename):
    with open(filename) as f:
        lines = f.readlines()
    f.close()
    return len(lines)


def write_folds(filename, dti_list):
    f = open(filename, "w")
    f.write(str(dti_list))
    f.close()


def get_active_inactive_index_lst(filename, flag, train_indexes):
    index_lst = []
    c = 0
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            comp_targ_lst = line.rstrip('\n').split("\t")
            if comp_targ_lst[2] == flag and c in train_indexes:
                index_lst.append(c)
            c += 1
    f.close()
    return index_lst


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    training_dataset_path = "{}/{}".format(training_files_path, args.d)

    training_dataset_size = read_file_return_total_line(training_dataset_path + "/comp_targ_binary.tsv")
    folds = json.load(open(training_dataset_path + "/data/folds/train_fold_setting1.txt"))
    test_indices = json.load(open(training_dataset_path + "/data/folds/test_fold_setting1.txt"))

    train_indices = []
    for fold_id in range(len(folds)):
        folds_id_list = list(range(len(folds)))
        for tr_fold_in in folds_id_list:
            train_indices.extend(folds[tr_fold_in])

    os.makedirs(os.path.dirname(training_dataset_path + "/dataSubset" + str(args.ss) + "/folds/"), exist_ok=True)
    train_fold_list = []

    sub_list_size = int(int(args.ss) / 2)

    active_training_dataset_lst = get_active_inactive_index_lst(training_dataset_path + "/comp_targ_binary.tsv", "1",
                                                                train_indices)
    inactive_training_dataset_lst = get_active_inactive_index_lst(training_dataset_path + "/comp_targ_binary.tsv", "0",
                                                                  train_indices)

    active_test_dataset_lst = get_active_inactive_index_lst(training_dataset_path + "/comp_targ_binary.tsv", "1",
                                                            test_indices)
    inactive_test_dataset_lst = get_active_inactive_index_lst(training_dataset_path + "/comp_targ_binary.tsv", "0",
                                                              test_indices)

    if sub_list_size > len(active_training_dataset_lst) or sub_list_size > len(inactive_training_dataset_lst):
        active_lst = active_training_dataset_lst + active_test_dataset_lst
        inactive_lst = inactive_training_dataset_lst + inactive_test_dataset_lst
        active_sub_List = random.sample(active_lst, sub_list_size)
        in_active_sub_List = random.sample(inactive_lst, sub_list_size)
    else:
        active_sub_List = random.sample(active_training_dataset_lst, sub_list_size)
        in_active_sub_List = random.sample(inactive_training_dataset_lst, sub_list_size)

    train_fold_list.extend(active_sub_List)
    train_fold_list.extend(in_active_sub_List)

    write_folds(training_dataset_path + "/dataSubset" + str(args.ss) + "/folds/train_fold_setting1.txt",
                train_fold_list)
