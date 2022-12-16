from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
import warnings
from evaluation_metrics import prec_rec_f1_acc_mcc, get_list_of_scores
import argparse
from sklearn.metrics import roc_curve

import json
import os
import pandas as pd
import numpy as np

warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser(description='baseline training arguments')

parser.add_argument(
    '--en',
    type=str,
    default="my_experiments",
    metavar='EN',
    help='the name of the experiment (default: my_experiment)')

parser.add_argument(
    # '--compound-features',
    '--cf',
    type=str,
    default="chemprop",
    metavar='CF',
    help='compound features separated by underscore character (default: chemprop)')

parser.add_argument(
    '--sd',
    type=str,
    default="kinase",
    metavar='SD',
    help='the name of the source dataset (default: kinase)')

parser.add_argument(
    '--td',
    type=str,
    default="transporter",
    metavar='TD',
    help='the name of the target dataset (default: transporter)')

parser.add_argument(
    '--ss',
    type=int,
    default=10,
    metavar='SS',
    help='subset size (default: 10)')

parser.add_argument(
    # '--extracted-layer`',
    '--el',
    type=str,
    default="0",
    metavar='EL',
    help='layer to be extracted (default: 1)')

parser.add_argument(
    # '--subset-flag',
    '--sf',
    type=int,
    default=0,
    metavar='SF',
    help='subset flag (default: 0)')

parser.add_argument(
    # '--transfer-learning-flag',
    '--tlf',
    type=int,
    default=0,
    metavar='TLF',
    help='transfer learning flag (default: 0)')

parser.add_argument(
    # '--setting',
    '--setting',
    type=int,
    default=1,
    metavar='SETTING',
    help='Determines the setting (1: train_val_test, 2:train_test) (default: 1)')

cwd = os.getcwd()
project_file_path = "{}TransferLearning4DTI".format(cwd.split("TransferLearning4DTI")[0])
training_files_path = "{}TransferLearning4DTI/training_files".format(cwd.split("TransferLearning4DTI")[0])
result_files_path = "{}/{}".format(project_file_path, "result_files/")


def read_tsv(file_name):
    feature_class = []
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")
            feature_class.append(line.split("\t")[1:])
    x = np.array(feature_class)
    y = x.astype(float)
    f.close()
    return y


def read_bioactivity_tsv(file_name):
    tar_comp_class = []
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")
            tar_comp_class.append(line.split("\t")[:])
    f.close()
    return tar_comp_class


def get_feature_list(feature_class):
    features = []
    for feature in feature_class:
        features.append(feature[0:])
    return features


def get_class_list(feature_class):
    classes = []
    for c in feature_class:
        classes.append(c[-1])
    return classes


def get_compound_dict_feature_vector(feature_lst):
    comp_feature_vector_path = "{}/compound_feature_vectors".format(training_dataset_path)
    feat_vec_path = comp_feature_vector_path
    comp_features_dict = dict()
    feature_fl_path = "{}/{}.tsv".format(feat_vec_path, feature_lst[0])
    with open(feature_fl_path) as f:
        for line in f:
            line = line.split("\n")[0]
            line = line.split("\t")
            compound_id = line[0]
            feat_vec = line[1:]
            comp_features_dict[compound_id] = feat_vec

    return comp_features_dict


def find_optimal_cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    indices = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1-fpr), index=indices), 'threshold': pd.Series(threshold, index=indices)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])


if __name__ == '__main__':

    args = parser.parse_args()

    subset_size = args.ss
    subset_flag = args.sf
    extracted_layer = args.el
    experiment_name = args.en
    source_dataset = args.sd
    target_dataset = args.td
    tl_flag = args.tlf
    setting = args.setting
    comp_feature = args.cf

    arguments = [str(argm) for argm in [source_dataset, target_dataset, comp_feature, experiment_name, subset_flag, extracted_layer, subset_size, setting,
                                        tl_flag]]
    str_arguments = "-".join(arguments)
    print("Arguments:", str_arguments)

    training_dataset_path = "{}/{}".format(training_files_path, target_dataset)

    if subset_flag == 0:
        exp_path = os.path.join(result_files_path, target_dataset)
    else:
        exp_path = os.path.join(result_files_path, target_dataset + "/dataSubset" + str(subset_size))

    exp_path = "{}/{}".format(exp_path, "/shallow")
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    svm_best_val_test_result_fl = open(
        "{}/svm_layer_{}_per_results-{}.txt".format(exp_path, extracted_layer, str_arguments), "w")
    rf_best_val_test_result_fl = open(
        "{}/rf_layer_{}_per_results-{}.txt".format(exp_path, extracted_layer, str_arguments), "w")
    gb_best_val_test_result_fl = open(
        "{}/gb_layer_{}_per_results-{}.txt".format(exp_path, extracted_layer, str_arguments), "w")

    folds = list()
    if tl_flag == 0:
        if subset_flag == 0:
            folds = json.load(open(training_dataset_path + "/data/folds/train_fold_setting1.txt"))
        else:
            folds = json.load(
                open(training_dataset_path + "/dataSubset" + str(subset_size) + "/folds/train_fold_setting1.txt"))

    comp_feature_list = ["chemprop"]
    if comp_feature == "ecfp4":
        comp_feature_list = ["ecfp4"]

    features_dict = get_compound_dict_feature_vector(comp_feature_list)
    bioactiviy_dataset = read_bioactivity_tsv(training_dataset_path + "/comp_targ_binary.tsv")
    test_indices = json.load(open(training_dataset_path + "/data/folds/test_fold_setting1.txt"))

    if setting == 1:
        avg_svm_val_mcc, avg_rf_val_mcc, avg_gb_val_mcc, avg_svm_test_mcc, avg_rf_test_mcc, avg_gb_test_mcc = 0, 0, 0, 0, 0, 0
        for fold in range(5):
            print("FOLD:", fold + 1)

            if tl_flag == 1:
                if subset_flag == 0:
                    features_path = training_dataset_path + "/extracted_feature_vectors/layer" + \
                                    extracted_layer + "/fold" + str(fold)
                else:
                    features_path = training_dataset_path + "/dataSubset" + str(subset_size) + \
                                "/extracted_feature_vectors/layer" + extracted_layer + "/fold" + str(fold)

                X_train = np.loadtxt(features_path + "/train.out", delimiter=',')
                y_train = np.loadtxt(features_path + "/trainClass.out", delimiter=',')

                X_val = np.loadtxt(features_path + "/val.out", delimiter=',')
                y_val = np.loadtxt(features_path + "/valClass.out", delimiter=',')

                X_test = np.loadtxt(features_path + "/test.out", delimiter=',')
                y_test = np.loadtxt(features_path + "/testClass.out", delimiter=',')

            else:
                X_val, y_val = [], []
                for ind in folds[fold]:
                    X_val.append(features_dict[bioactiviy_dataset[ind][1]])
                    y_val.append(int(bioactiviy_dataset[ind][2]))

                X_train, y_train = [], []
                for j in range(5):
                    if fold == j:
                        continue

                    else:
                        for ind in folds[j]:
                            X_train.append(features_dict[bioactiviy_dataset[ind][1]])
                            y_train.append(int(bioactiviy_dataset[ind][2]))

                X_test, y_test = [], []
                for ind in test_indices:
                    X_test.append(features_dict[bioactiviy_dataset[ind][1]])
                    y_test.append(int(bioactiviy_dataset[ind][2]))

            svm_classifier = svm.SVC()  # Linear Kernel
            print("SVM Training")
            # Train the model using the training sets
            svm_classifier.fit(X_train, y_train)
            print("SVM Validate")

            # Predict the response for test dataset
            y_pred = svm_classifier.predict(X_val)
            svm_val_perf_dict = prec_rec_f1_acc_mcc(y_val, y_pred, 2)
            print(svm_val_perf_dict)
            print("SVM Test")

            y_pred = svm_classifier.predict(X_test)
            svm_test_perf_dict = prec_rec_f1_acc_mcc(y_test, y_pred, 2)
            print(svm_test_perf_dict)

            rf_classifier = RandomForestClassifier()  # Linear Kernel
            print("RF Training")

            # Train the model using the training sets
            rf_classifier.fit(X_train, y_train)
            print("RF Validate")

            # Predict the response for test dataset
            y_pred = rf_classifier.predict(X_val)
            rf_val_perf_dict = prec_rec_f1_acc_mcc(y_val, y_pred, 2)
            print(rf_val_perf_dict)
            print("RF Test")
            y_pred = rf_classifier.predict(X_test)
            rf_test_perf_dict = prec_rec_f1_acc_mcc(y_test, y_pred, 2)
            print(rf_test_perf_dict)

            gb_classifier = GradientBoostingClassifier()  # Linear Kernel
            print("GB Training")

            # Train the model using the training sets
            gb_classifier.fit(X_train, y_train)
            print("GB Validate")

            # Predict the response for test dataset
            y_pred = gb_classifier.predict(X_val)
            gb_val_perf_dict = prec_rec_f1_acc_mcc(y_val, y_pred, 2)
            print(gb_val_perf_dict)

            print("GB Test")
            y_pred = gb_classifier.predict(X_test)
            gb_test_perf_dict = prec_rec_f1_acc_mcc(y_test, y_pred, 2)
            print(gb_test_perf_dict)

            avg_svm_val_mcc += svm_val_perf_dict["MCC"]
            avg_svm_test_mcc += svm_test_perf_dict["MCC"]

            avg_rf_val_mcc += rf_val_perf_dict["MCC"]
            avg_rf_test_mcc += rf_test_perf_dict["MCC"]

            avg_gb_val_mcc += gb_val_perf_dict["MCC"]
            avg_gb_test_mcc += gb_test_perf_dict["MCC"]

            score_list = get_list_of_scores(2)
            for scr in score_list:
                svm_best_val_test_result_fl.write("Val {}:\t{}\n".format(scr, svm_val_perf_dict[scr]))
            for scr in score_list:
                svm_best_val_test_result_fl.write("Test {}:\t{}\n".format(scr, svm_test_perf_dict[scr]))
            for scr in score_list:
                rf_best_val_test_result_fl.write("Val {}:\t{}\n".format(scr, rf_val_perf_dict[scr]))
            for scr in score_list:
                rf_best_val_test_result_fl.write("Test {}:\t{}\n".format(scr, rf_test_perf_dict[scr]))
            for scr in score_list:
                gb_best_val_test_result_fl.write("Val {}:\t{}\n".format(scr, gb_val_perf_dict[scr]))
            for scr in score_list:
                gb_best_val_test_result_fl.write("Test {}:\t{}\n".format(scr, gb_test_perf_dict[scr]))
            if fold == 4:
                avg_svm_val_mcc /= 5
                avg_svm_test_mcc /= 5

                avg_rf_val_mcc /= 5
                avg_rf_test_mcc /= 5

                avg_gb_val_mcc /= 5
                avg_gb_test_mcc /= 5

                svm_best_val_test_result_fl.write("Val avg mcc:\t{}\n".format(avg_svm_val_mcc))
                svm_best_val_test_result_fl.write("Test avg mcc:\t{}\n".format(avg_svm_test_mcc))

                rf_best_val_test_result_fl.write("Val avg mcc:\t{}\n".format(avg_rf_val_mcc))
                rf_best_val_test_result_fl.write("Test avg mcc:\t{}\n".format(avg_rf_test_mcc))

                gb_best_val_test_result_fl.write("Val avg mcc:\t{}\n".format(avg_gb_val_mcc))
                gb_best_val_test_result_fl.write("Test avg mcc:\t{}\n".format(avg_gb_test_mcc))

        svm_best_val_test_result_fl.close()
        rf_best_val_test_result_fl.close()
        gb_best_val_test_result_fl.close()

    if setting == 2:

        if tl_flag == 1:
            if subset_flag == 0:
                features_path = training_dataset_path + "/extracted_feature_vectors/layer" + \
                                extracted_layer
            else:
                features_path = training_dataset_path + "/dataSubset" + str(subset_size) + \
                            "/extracted_feature_vectors/layer" + extracted_layer

            X_train = np.loadtxt(features_path + "/train.out", delimiter=',')
            y_train = np.loadtxt(features_path + "/trainClass.out", delimiter=',')

            X_test = np.loadtxt(features_path + "/test.out", delimiter=',')
            y_test = np.loadtxt(features_path + "/testClass.out", delimiter=',')

        else:

            X_train, y_train = [], []
            if subset_flag == 0:
                for i in range(5):
                    for ind in folds[i]:
                        X_train.append(features_dict[bioactiviy_dataset[ind][1]])
                        y_train.append(int(bioactiviy_dataset[ind][2]))
            else:
                for ind in folds:
                    X_train.append(features_dict[bioactiviy_dataset[ind][1]])
                    y_train.append(int(bioactiviy_dataset[ind][2]))

            X_test, y_test = [], []
            for ind in test_indices:
                X_test.append(features_dict[bioactiviy_dataset[ind][1]])
                y_test.append(int(bioactiviy_dataset[ind][2]))

        svm_classifier = svm.SVC()  # Linear Kernel
        # Train the model using the training sets
        svm_classifier.fit(X_train, y_train)
        print("SVM Test")

        y_pred = svm_classifier.predict(X_test)
        svm_test_perf_dict = prec_rec_f1_acc_mcc(y_test, y_pred, 2)
        svm_threshold = find_optimal_cutoff(y_test, y_pred)
        svm = 0
        for pred in y_pred:
            if pred >= svm_threshold[0]:
                svm += 1
        print(svm_test_perf_dict)

        rf_classifier = RandomForestClassifier()  # Linear Kernel

        # Train the model using the training sets
        rf_classifier.fit(X_train, y_train)

        print("RF Test")
        y_pred = rf_classifier.predict(X_test)
        rf_test_perf_dict = prec_rec_f1_acc_mcc(y_test, y_pred, 2)
        rf_threshold = find_optimal_cutoff(y_test, y_pred)
        rf = 0
        for pred in y_pred:
            if pred >= rf_threshold[0]:
                rf += 1
        print(rf_test_perf_dict)

        gb_classifier = GradientBoostingClassifier()  # Linear Kernel

        # Train the model using the training sets
        gb_classifier.fit(X_train, y_train)

        print("GB Test")
        y_pred = gb_classifier.predict(X_test)
        gb_test_perf_dict = prec_rec_f1_acc_mcc(y_test, y_pred, 2)
        gb_threshold = find_optimal_cutoff(y_test, y_pred)
        gb = 0
        for pred in y_pred:
            if pred >= gb_threshold[0]:
                gb += 1
        print(gb_test_perf_dict)

        score_list = get_list_of_scores(2)

        for scr in score_list:
            svm_best_val_test_result_fl.write("Test {}:\t{}\n".format(scr, svm_test_perf_dict[scr]))

        for scr in score_list:
            rf_best_val_test_result_fl.write("Test {}:\t{}\n".format(scr, rf_test_perf_dict[scr]))

        for scr in score_list:
            gb_best_val_test_result_fl.write("Test {}:\t{}\n".format(scr, gb_test_perf_dict[scr]))

        svm_best_val_test_result_fl.write("Test mcc:\t{}\n".format(svm_test_perf_dict["MCC"]))

        rf_best_val_test_result_fl.write("Test mcc:\t{}\n".format(rf_test_perf_dict["MCC"]))

        gb_best_val_test_result_fl.write("Test mcc:\t{}\n".format(gb_test_perf_dict["MCC"]))

    svm_best_val_test_result_fl.close()
    rf_best_val_test_result_fl.close()
    gb_best_val_test_result_fl.close()
