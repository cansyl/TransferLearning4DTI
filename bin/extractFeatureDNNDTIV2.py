import os
import warnings
from models import get_model
from evaluation_metrics import prec_rec_f1_acc_mcc
from data_processing import get_test_val_folds_train_data_loader, get_train_test_train_data_loader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
warnings.filterwarnings(action='ignore')


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


def compute_test_loss(model, criterion, data_loader, device, extracted_layer, test_features, num_classes):
    total_count = 0
    total_loss = 0.0
    all_comp_ids = []
    all_tar_ids = []
    all_labels = []
    predictions = []
    total_number_test = 0
    for i, data in enumerate(data_loader):

        comp_feature_vectors, labels, compound_ids, target_ids = data
        comp_feature_vectors, labels = Variable(comp_feature_vectors).to(
            device), Variable(labels).to(device)
        all_comp_ids.extend(compound_ids)
        all_tar_ids.extend(target_ids)
        total_count += comp_feature_vectors.shape[0]

        if extracted_layer == "1":
            model.l1.register_forward_hook(get_activation('l1'))
            y_pred = model(comp_feature_vectors).to(device)
            extracted_features = np.array(activation['l1'].cpu())
        elif extracted_layer == "2":
            model.l2.register_forward_hook(get_activation('l2'))
            y_pred = model(comp_feature_vectors).to(device)
            extracted_features = np.array(activation['l2'].cpu())
        if total_number_test == 0:
            test_features = extracted_features

        else:
            test_features = np.append(test_features, extracted_features, axis=0)

        total_number_test += len(labels)
        if num_classes == 2:
            y_test_pred = torch.sigmoid(y_pred)
            y_pred_tag = torch.round(y_test_pred)
            loss_val = criterion(y_pred.squeeze(), labels)
            total_loss += float(loss_val.item())
            for item in labels:
                all_labels.append(float(item.item()))

            for item in y_pred_tag:
                predictions.append(float(item.item()))
        else:
            loss_val = criterion(y_pred.squeeze(), labels.long())
            total_loss += float(loss_val.item())
            for item in labels:
                all_labels.append(float(item.item()))

            _, y_pred_tag = torch.max(y_pred, dim=1)

            for item in y_pred_tag:
                predictions.append(float(item.item()))
    return total_loss, total_count, all_labels, predictions, all_comp_ids, all_tar_ids, test_features


activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def extract_features(target_dataset, source_dataset, comp_feature_list, comp_hidden_lst, learning_rate, batch_size,
                     model_nm, dropout, experiment_name, n_epoch, subset_flag, tl_flag, freeze_flag, freezing_layers,
                     extracted_layer, subset_size, input_path, setting):
    arguments = [str(argm) for argm in [target_dataset, source_dataset, comp_hidden_lst,  learning_rate, batch_size,
                                        experiment_name, model_nm, dropout, n_epoch, setting, subset_flag, tl_flag, freeze_flag,
                                        freezing_layers, extracted_layer, subset_size]]

    str_arguments = "-".join(arguments)
    print("Arguments:", str_arguments)

    torch.manual_seed(101)
    np.random.seed(101)
    training_files_path = "{}/{}".format(input_path, "training_files")
    training_dataset_path = "{}/{}".format(training_files_path, target_dataset)
    trained_models_path = "{}/{}".format(input_path, "trained_models/")

    loader_fold_dict, test_loader, external_data_loader = get_test_val_folds_train_data_loader(training_dataset_path,
                                                                                               comp_feature_list,
                                                                                               batch_size, subset_size,
                                                                                               subset_flag, setting)
    num_of_folds = len(loader_fold_dict)
    folds = range(num_of_folds)
    for fold in folds:
        total_number_train = 0

        train_features = np.empty((batch_size, comp_hidden_lst[int(extracted_layer) - 1]))
        val_features = np.empty((batch_size, comp_hidden_lst[int(extracted_layer) - 1]))
        test_features = np.empty((batch_size, comp_hidden_lst[int(extracted_layer) - 1]))
        allLabels = []
        # change here for setting 2
        if subset_flag == 1:
            features_path = training_dataset_path + "/dataSubset" + str(subset_size) + \
                        "/extracted_feature_vectors/layer" + extracted_layer + "/fold" + str(fold)
        else:
            features_path = training_dataset_path +  \
                            "/extracted_feature_vectors/layer" + extracted_layer + "/fold" + str(fold)
        if not os.path.exists(features_path):
            os.makedirs(features_path)
        print("FOLD:", fold + 1)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        train_loader, valid_loader = loader_fold_dict[fold]
        if comp_feature_list[0] == "ecfp4":
            comp_feature_size = 1024
        elif comp_feature_list[0] == "chemprop":
            comp_feature_size = 300

        model = get_model(model_nm, comp_feature_size, comp_hidden_lst, dropout)
        if tl_flag == 1:
            model.load_state_dict(torch.load("{}/{}/best_val-state_dict-fold-{}.pth".format(trained_models_path,
                                                                                             source_dataset, 0),
                                             map_location=torch.device(device)))
        model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        # freeze layers
        if "1" in freezing_layers:
            model.l1.bias.requires_grad = False
            model.l1.weight.requires_grad = False
        if "2" in freezing_layers:
            model.l2.bias.requires_grad = False
            model.l2.weight.requires_grad = False
        if "3" in freezing_layers:
            model.l3.bias.requires_grad = False
            model.l3.weight.requires_grad = False

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        best_val_test_performance_dict = dict()
        best_val_test_performance_dict["MCC"] = 0.0
        model.train()
        for epoch in range(1, n_epoch + 1):
            total_training_loss, total_validation_loss, total_test_loss = 0.0, 0.0, 0.0
            epoch_loss = 0
            epoch_acc = 0

            for i, data in enumerate(train_loader):
                optimizer.zero_grad()


                comp_feature_vectors, labels, compound_ids, target_ids = data
                comp_feature_vectors, labels = Variable(comp_feature_vectors).to(
                    device), Variable(labels).to(device)

                y_pred = model(comp_feature_vectors).to(device)

                loss = criterion(y_pred.squeeze(), labels)
                total_training_loss += float(loss.item())
                epoch_loss += loss.item()
                acc = binary_acc(y_pred.squeeze(), labels)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += acc.item()
                if epoch == n_epoch:


                    if extracted_layer == "1":
                        model.l1.register_forward_hook(get_activation('l1'))
                        y_pred = model(comp_feature_vectors).to(device)
                        extractedFeatures = np.array(activation['l1'].cpu())
                    elif extracted_layer == "2":
                        model.l2.register_forward_hook(get_activation('l2'))
                        y_pred = model(comp_feature_vectors).to(device)
                        extractedFeatures = np.array(activation['l2'].cpu())
                    if total_number_train == 0:
                        train_features = extractedFeatures
                        allLabels += labels.cpu()
                    else:
                        train_features = np.append(train_features, extractedFeatures, axis=0)
                        allLabels += labels.cpu()

                    total_number_train += 1
                        # print(trainFeatures.shape)
                    # save extracted features
                    if total_number_train == len(train_loader):
                        np.savetxt(features_path + "/train.out", train_features, delimiter=',', fmt='%1.6f')
                        np.savetxt(features_path + "/trainClass.out", allLabels, delimiter='\n', fmt='%1.6f')

            model.eval()
            with torch.no_grad():  # torch.set_grad_enabled(False):
                total_validation_loss, total_validation_count, validation_labels, validation_predictions, \
                all_val_comp_ids, all_val_tar_ids, val_features = compute_test_loss(
                    model, criterion, valid_loader, device, extracted_layer, val_features)

                total_test_loss, total_test_count, test_labels, test_predictions, all_test_comp_ids, \
                all_test_tar_ids, test_features = compute_test_loss(
                    model, criterion, test_loader, device, extracted_layer, test_features)

            val_perf_dict = dict()
            val_perf_dict["MCC"] = 0.0

            val_perf_dict = prec_rec_f1_acc_mcc(validation_labels, validation_predictions)

            test_perf_dict = dict()
            test_perf_dict["MCC"] = 0.0

            test_perf_dict = prec_rec_f1_acc_mcc(test_labels, test_predictions)
            print(f'Epoch {epoch + 0:03}: | Loss: {total_training_loss:.5f}  | Val_loss: {total_validation_loss:.5f} '
                  f'| Acc: {epoch_acc / len(train_loader):.3f} | Val_MCC: {val_perf_dict["MCC"]:.4f} '
                  f'| Test_MCC: {test_perf_dict["MCC"]:.4f}')
            # print("-----------------")
            # print(validation_labels)
            # print(test_labels)
            # print("-----------------")

            if epoch == n_epoch:
                # save extracted features

                np.savetxt(features_path + "/val.out", val_features, delimiter=',', fmt='%1.6f')
                np.savetxt(features_path + "/valClass.out", validation_labels, delimiter='\n', fmt='%1.6f')



                np.savetxt(features_path + "/test.out", test_features, delimiter=',', fmt='%1.6f')
                np.savetxt(features_path + "/testClass.out", test_labels, delimiter='\n', fmt='%1.6f')

def extract_features_train_test(target_dataset, source_dataset, comp_feature_list, comp_hidden_lst, learning_rate, batch_size,
                     model_nm, dropout, experiment_name, n_epoch, subset_flag, tl_flag, freeze_flag, freezing_layers,
                     extracted_layer, subset_size, input_path, setting, num_classes):
    arguments = [str(argm) for argm in [target_dataset, source_dataset, comp_hidden_lst,  learning_rate, batch_size,
                                        experiment_name, model_nm, dropout, n_epoch, setting, subset_flag, tl_flag, freeze_flag,
                                        freezing_layers, extracted_layer, subset_size, num_classes]]


    str_arguments = "-".join(arguments)
    print("Arguments:", str_arguments)

    torch.manual_seed(101)
    np.random.seed(101)
    training_files_path = "{}/{}".format(input_path, "training_files")
    training_dataset_path = "{}/{}".format(training_files_path, target_dataset)
    trained_models_path = "{}/{}".format(input_path, "trained_models/")



    train_loader, test_loader, external_data_loader = get_train_test_train_data_loader(training_dataset_path, comp_feature_list,
                                                                                       batch_size, subset_size, subset_flag, setting)

    total_number_train = 0


    trainFeatures = np.empty((batch_size, comp_hidden_lst[int(extracted_layer) - 1]))
    testFeatures = np.empty((batch_size, comp_hidden_lst[int(extracted_layer) - 1]))
    allLabels = []
    # change here for setting 2
    if subset_flag == 1:
        features_path = training_dataset_path + "/dataSubset" + str(subset_size) + \
                    "/extracted_feature_vectors/layer" + extracted_layer
    else:
        features_path = training_dataset_path +  \
                        "/extracted_feature_vectors/layer" + extracted_layer
    if not os.path.exists(features_path):
        os.makedirs(features_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if comp_feature_list[0] == "ecfp4":
        comp_feature_size = 1024
    elif comp_feature_list[0] == "chemprop":
        comp_feature_size = 300

    model = get_model(model_nm, comp_feature_size, comp_hidden_lst, num_classes, dropout)
    if tl_flag == 1:
        model.load_state_dict(torch.load("{}/{}/best_val-state_dict-fold-{}.pth".format(trained_models_path,
                                                                                         source_dataset, 5),
                                         map_location=torch.device(device)))
    model.to(device)
    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    # freeze layers

    if "1" in freezing_layers:
        model.l1.bias.requires_grad = False
        model.l1.weight.requires_grad = False
    if "2" in freezing_layers:
        model.l2.bias.requires_grad = False
        model.l2.weight.requires_grad = False
    if "3" in freezing_layers:
        model.l3.bias.requires_grad = False
        model.l3.weight.requires_grad = False

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    best_val_test_performance_dict = dict()
    best_val_test_performance_dict["MCC"] = 0.0
    model.train()
    for epoch in range(1, n_epoch + 1):
        total_training_loss, total_test_loss = 0.0, 0.0
        epoch_loss = 0
        epoch_acc = 0

        for i, data in enumerate(train_loader):
            optimizer.zero_grad()


            comp_feature_vectors, labels, compound_ids, target_ids = data
            comp_feature_vectors, labels = Variable(comp_feature_vectors).to(
                device), Variable(labels).to(device)

            y_pred = model(comp_feature_vectors).to(device)

            if num_classes == 2:
                loss = criterion(y_pred.squeeze(), labels)
            else:
                loss = criterion(y_pred.squeeze(), labels.long())
            total_training_loss += float(loss.item())
            epoch_loss += loss.item()
            if num_classes == 2:
                acc = binary_acc(y_pred.squeeze(), labels)
            else:
                acc = multi_acc(y_pred.squeeze(), labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            if epoch == n_epoch:

                if extracted_layer == "1":
                    model.l1.register_forward_hook(get_activation('l1'))
                    y_pred = model(comp_feature_vectors).to(device)
                    extractedFeatures = np.array(activation['l1'].cpu())
                elif extracted_layer == "2":
                    model.l2.register_forward_hook(get_activation('l2'))
                    y_pred = model(comp_feature_vectors).to(device)
                    extractedFeatures = np.array(activation['l2'].cpu())
                if total_number_train == 0:
                    trainFeatures = extractedFeatures
                    allLabels += labels.cpu()
                else:
                    trainFeatures = np.append(trainFeatures, extractedFeatures, axis=0)
                    allLabels += labels.cpu()

                total_number_train += 1
                    # print(trainFeatures.shape)
                # save extracted features
                if total_number_train == len(train_loader):
                    np.savetxt(features_path + "/train.out", trainFeatures, delimiter=',', fmt='%1.6f')
                    np.savetxt(features_path + "/trainClass.out", allLabels, delimiter='\n', fmt='%1.6f')

        model.eval()
        with torch.no_grad():  # torch.set_grad_enabled(False):
            total_test_loss, total_test_count, test_labels, test_predictions, all_test_comp_ids, \
            all_test_tar_ids, testFeatures = compute_test_loss(
                model, criterion, test_loader, device, extracted_layer, testFeatures, num_classes)

        test_perf_dict = dict()
        test_perf_dict["MCC"] = 0.0

        test_perf_dict = prec_rec_f1_acc_mcc(test_labels, test_predictions, num_classes)
        print(f'Epoch {epoch + 0:03}: | Loss: {total_training_loss:.5f}  '
              f'| Acc: {epoch_acc / len(train_loader):.3f} '
              f'| Test_MCC: {test_perf_dict["MCC"]:.4f}')


        if epoch == n_epoch:
            # save extracted features
            np.savetxt(features_path + "/test.out", testFeatures, delimiter=',', fmt='%1.6f')
            np.savetxt(features_path + "/testClass.out", test_labels, delimiter='\n', fmt='%1.6f')
