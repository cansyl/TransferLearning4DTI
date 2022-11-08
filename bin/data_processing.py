import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from rdkit import Chem
from rdkit.Chem import AllChem
import json


def get_ecfp4_features_given_smiles_dict(smiles_dict):

    compound_smile_dict = {}
    for comp_id in smiles_dict:
        m = Chem.MolFromSmiles(smiles_dict[comp_id])
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
        feature_list = []
        for dim in fp:
            feature_list.append(float(dim))
        compound_smile_dict[comp_id] = feature_list

    return compound_smile_dict


def read_smiles(file_name):
    compound_smile_dict = {}
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            compound_smile_pair = line.rstrip('\n').split("\t")
            compound_smile_dict[compound_smile_pair[0]] = compound_smile_pair[1]

    return compound_smile_dict


def get_target_dict_feature_vector(training_dataset_path, feature_lst):
    tar_feature_vector_path = "{}/target_feature_vectors".format(training_dataset_path)
    feat_vec_path = tar_feature_vector_path
    features_dict = dict()
    feature_fl_path = "{}/{}.tsv".format(feat_vec_path, feature_lst[0])
    with open(feature_fl_path) as f:
        for line in f:
            line = line.split("\n")[0]
            line = line.split("\t")
            target_id = line[0]
            feat_vec = line[1:]
            features_dict[target_id] = torch.tensor(np.asarray(feat_vec, dtype=float)).type(torch.FloatTensor)
    return features_dict


def get_compound_dict_feature_vector(training_dataset_path, feature_lst):
    comp_feature_vector_path = "{}/compound_feature_vectors".format(training_dataset_path)
    feat_vec_path = comp_feature_vector_path
    features_dict = dict()
    feature_fl_path = "{}/{}.tsv".format(feat_vec_path, feature_lst[0])

    with open(feature_fl_path) as f:
        for line in f:
            line = line.split("\n")[0]
            line = line.split("\t")
            compound_id = line[0]
            feat_vec = line[1:]
            features_dict[compound_id] = torch.tensor(np.asarray(feat_vec, dtype=float)).type(torch.FloatTensor)

    return features_dict


def get_test_compound_dict_feature_vector(training_dataset_path, feature_lst):
    if feature_lst[0] == "chemprop":
        print(feature_lst)
        features_dict = dict()
        feature_fl_path = training_dataset_path
        with open(feature_fl_path) as f:
            compound_list = []
            for line in f:
                line = line.split("\n")[0]
                line = line.split("\t")
                compound_id = line[0]
                feat_vec = line[1:]
                compound_list.append(compound_id)
                features_dict[compound_id] = torch.tensor(np.asarray(feat_vec, dtype=float)).type(torch.FloatTensor)
        return features_dict, compound_list
    elif feature_lst[0] == "ecfp4":
        compound_smile_dict = read_smiles(training_dataset_path)
        compound_features = get_ecfp4_features_given_smiles_dict(compound_smile_dict)
        features_dict = dict()
        compound_list = []
        for compound, feat_vec in compound_features.items():
            features_dict[compound] = torch.tensor(np.asarray(feat_vec, dtype=float)).type(torch.FloatTensor)
            compound_list.append(compound)

        return features_dict, compound_list


class BioactivityDataset(Dataset):
    def __init__(self, training_dataset_path, comp_target_pair_dataset, compound_feature_list):
        self.training_dataset_path = training_dataset_path
        self.compound_feature_list = compound_feature_list
        comp_target_pair_dataset_path = "{}/{}".format(training_dataset_path, comp_target_pair_dataset)

        self.dict_compound_features = get_compound_dict_feature_vector(training_dataset_path, compound_feature_list)

        self.training_dataset = pd.read_csv(comp_target_pair_dataset_path, header=None, sep='\t')
        self.num_of_train_test_val = len(self.training_dataset)

    def get_num_of_train_test_val(self):
        return self.num_of_train_test_val

    def __len__(self):
        return len(self.training_dataset)

    def __getitem__(self, idx):
        row = self.training_dataset.iloc[idx]
        tar_id, comp_id, bio_act_val = str(row[0]), str(row[1]), str(row[2])
        comp_feats = self.dict_compound_features[comp_id]
        label = torch.tensor(float(bio_act_val)).type(torch.FloatTensor)

        return comp_feats, label, comp_id, tar_id


class BioactivityTestDataset(Dataset):
    def __init__(self, training_dataset_path, compound_feature_list):
        self.training_dataset_path = training_dataset_path
        self.compound_feature_list = compound_feature_list
        self.dict_compound_features, self.compound_list = get_test_compound_dict_feature_vector(training_dataset_path,
                                                                                                compound_feature_list)

    def __len__(self):
        return len(self.dict_compound_features)

    def __getitem__(self, idx):
        comp_id = self.compound_list[idx]
        comp_feats = self.dict_compound_features[comp_id]

        return comp_feats, comp_id


def get_test_val_folds_train_data_loader(training_dataset_path, comp_feature_list, batch_size, subset_size, subset_flag):
    compound_target_pair_dataset = "comp_targ_binary.tsv"
    if subset_flag == 0:
        folds = json.load(open(training_dataset_path + "/data/folds/train_fold_setting1.txt"))
    else:
        folds = json.load(
            open(training_dataset_path + "/dataSubset" + str(subset_size) + "/folds/train_fold_setting1.txt"))
    test_indices = json.load(open(training_dataset_path + "/data/folds/test_fold_setting1.txt"))

    bioactivity_dataset = BioactivityDataset(training_dataset_path, compound_target_pair_dataset, comp_feature_list)
    loader_fold_dict = dict()
    for fold_id in range(len(folds)):
        folds_id_list = list(range(len(folds)))
        val_indices = folds[fold_id]
        folds_id_list.remove(fold_id)
        train_indices = []
        for tr_fold_in in folds_id_list:
            train_indices.extend(folds[tr_fold_in])
        train_indices = train_indices  # [:10]
        val_indices = val_indices  # [:10]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(bioactivity_dataset, batch_size=batch_size,
                                                   sampler=train_sampler)
        valid_loader = torch.utils.data.DataLoader(bioactivity_dataset, batch_size=batch_size,
                                                   sampler=valid_sampler)

        loader_fold_dict[fold_id] = [train_loader, valid_loader]

    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = torch.utils.data.DataLoader(bioactivity_dataset, batch_size=batch_size,
                                              sampler=test_sampler)
    external_test_loader = None

    return loader_fold_dict, test_loader, external_test_loader


def get_train_data_loader(training_dataset_path, comp_feature_list, batch_size, subset_size, subset_flag,
                          external_test):
    import json

    compound_target_pair_dataset = "comp_targ_binary.tsv"
    if subset_flag == 0:
        folds = json.load(open(training_dataset_path + "/data/folds/train_fold_setting1.txt"))
    else:
        folds = json.load(
            open(training_dataset_path + "/dataSubset" + str(subset_size) + "/folds/train_fold_setting1.txt"))
    test_indices = json.load(open(training_dataset_path + "/data/folds/test_fold_setting1.txt"))

    bioactivity_dataset = BioactivityDataset(training_dataset_path, compound_target_pair_dataset, comp_feature_list)
    train_indices = []
    for fold_id in range(len(folds)):
        train_indices.extend(folds[fold_id])
    train_indices.extend(test_indices)

    train_sampler = SequentialSampler(train_indices)

    train_loader = torch.utils.data.DataLoader(bioactivity_dataset, batch_size=batch_size,
                                               sampler=train_sampler)

    external_test_loader = None
    if external_test != "-":
        bioactivity_dataset = BioactivityTestDataset(external_test, comp_feature_list)
        external_indices = []
        for i in range(bioactivity_dataset.__len__()):
            external_indices.append(i)
        external_sampler = SequentialSampler(external_indices)
        external_test_loader = torch.utils.data.DataLoader(bioactivity_dataset, batch_size=batch_size,
                                                           sampler=external_sampler)
    return train_loader, external_test_loader


def get_train_test_train_data_loader(training_dataset_path, comp_feature_list, batch_size, subset_size, subset_flag):

    compound_target_pair_dataset = "comp_targ_binary.tsv"
    if subset_flag == 0:
        folds = json.load(open(training_dataset_path + "/data/folds/train_fold_setting1.txt"))
    else:
        folds = json.load(
            open(training_dataset_path + "/dataSubset" + str(subset_size) + "/folds/train_fold_setting1.txt"))
    test_indices = json.load(open(training_dataset_path + "/data/folds/test_fold_setting1.txt"))

    bioactivity_dataset = BioactivityDataset(training_dataset_path, compound_target_pair_dataset, comp_feature_list)
    train_indices = []
    if subset_flag == 0:
        for fold_id in range(len(folds)):
            train_indices.extend(folds[fold_id])
    else:
        train_indices = folds

    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = torch.utils.data.DataLoader(bioactivity_dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = torch.utils.data.DataLoader(bioactivity_dataset, batch_size=batch_size,
                                              sampler=test_sampler)
    external_test_loader = None

    return train_loader, test_loader, external_test_loader
