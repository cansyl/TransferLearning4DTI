
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
import warnings
from evaluation_metrics import prec_rec_f1_acc_mcc, get_list_of_scores
import argparse


import json
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser(description='create training and test datasets arguments')

parser.add_argument(
    '--en',
    type=str,
    default="my_experiments",
    metavar='EN',
    help='the name of the experiment (default: my_experiment)')

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

parser.add_argument(
    # '--extracted-layer`',
    '--el',
    type=str,
    default="1",
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
    # '--input-path`',
    '--ip',
    type=str,
    default="/home/adalkiran/PycharmProjects/mainProteinFamilyClassification",
    metavar='IP',
    help='input path (default: /home/adalkiran/PycharmProjects/mainProteinFamilyClassification)')

def read_tsv(FileName):
    featuresandClass = []
    with open(FileName) as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")
            featuresandClass.append(line.split("\t")[1:])
            # print(line.split("\t")[0:])
    x = np.array(featuresandClass)
    y = x.astype(float)
    f.close()
    return y
def read_bioactivity_tsv(FileName):
    tar_comp_class = []
    with open(FileName) as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")
            tar_comp_class.append(line.split("\t")[:])
    f.close()
    return tar_comp_class

def getFeatureList(featuresandClass):
    features = []
    for feature in featuresandClass:
        features.append(feature[0:])
    return features

def getClassList(featuresandClass):
    classes = []
    for c in featuresandClass:
        classes.append(c[-1])
    return classes

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
            features_dict[compound_id] = feat_vec

    return features_dict

if __name__ == '__main__':

    args = parser.parse_args()

    subset_size = args.ss
    subset_flag = args.sf
    input_path = args.ip
    extracted_layer = args.el
    experiment_name = args.en
    target_dataset = args.d
    tl_flag = args.tlf

    arguments = [str(argm) for argm in [target_dataset, experiment_name, subset_flag, extracted_layer, subset_size]]
    str_arguments = "-".join(arguments)
    print("Arguments:", str_arguments)

    training_files_path = "{}/{}".format(input_path, "training_files")
    training_dataset_path = "{}/{}".format(training_files_path, target_dataset)
    result_files_path = "{}/{}".format(input_path, "result_files")

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

    if tl_flag == 0:
        if subset_flag == 0:
            folds = json.load(open(training_dataset_path + "/data/folds/train_fold_setting1.txt"))
        else:
            folds = json.load(
                open(training_dataset_path + "/dataSubset" + str(subset_size) + "/folds/train_fold_setting1.txt"))
        comp_feature_list = ["ecfp4"]
        features_dict = get_compound_dict_feature_vector(training_dataset_path, comp_feature_list)
        bioactiviy_dataset = read_bioactivity_tsv(training_dataset_path + "/comp_targ_binary.tsv")
        test_indices = json.load(open(training_dataset_path + "/data/folds/test_fold_setting1.txt"))

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

        # print(len(X_train))
        # print(len(X_val))
        # print(len(X_test))
        # print(classes)
        svm_classifier = svm.SVC()  # Linear Kernel
        print("SVM Training")
        # Train the model using the training sets
        svm_classifier.fit(X_train, y_train)
        print("SVM Training End")
        print("SVM Validate")

        # Predict the response for test dataset
        y_pred = svm_classifier.predict(X_val)
        svm_val_perf_dict = prec_rec_f1_acc_mcc(y_val, y_pred)
        print(svm_val_perf_dict)
        print("SVM Test")

        y_pred = svm_classifier.predict(X_test)
        svm_test_perf_dict = prec_rec_f1_acc_mcc(y_test, y_pred)
        print(svm_test_perf_dict)

        rf_classifier = RandomForestClassifier()  # Linear Kernel
        print("RF Training")

        # Train the model using the training sets
        rf_classifier.fit(X_train, y_train)
        print("RF Validate")

        # Predict the response for test dataset
        y_pred = rf_classifier.predict(X_val)
        rf_val_perf_dict = prec_rec_f1_acc_mcc(y_val, y_pred)
        print(rf_val_perf_dict)
        print("RF Test")
        y_pred = rf_classifier.predict(X_test)
        rf_test_perf_dict = prec_rec_f1_acc_mcc(y_test, y_pred)
        print(rf_test_perf_dict)

        gb_classifier = GradientBoostingClassifier()  # Linear Kernel
        print("GB Training")

        # Train the model using the training sets
        gb_classifier.fit(X_train, y_train)
        print("GB Validate")

        # Predict the response for test dataset
        y_pred = gb_classifier.predict(X_val)
        gb_val_perf_dict = prec_rec_f1_acc_mcc(y_val, y_pred)
        print(gb_val_perf_dict)

        print("GB Test")
        y_pred = gb_classifier.predict(X_test)
        gb_test_perf_dict = prec_rec_f1_acc_mcc(y_test, y_pred)
        print(gb_test_perf_dict)

        avg_svm_val_mcc += svm_val_perf_dict["MCC"]
        avg_svm_test_mcc += svm_test_perf_dict["MCC"]

        avg_rf_val_mcc += rf_val_perf_dict["MCC"]
        avg_rf_test_mcc += rf_test_perf_dict["MCC"]

        avg_gb_val_mcc += gb_val_perf_dict["MCC"]
        avg_gb_test_mcc += gb_test_perf_dict["MCC"]

        score_list = get_list_of_scores()
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

