import os
import warnings
from models import get_model
from evaluation_metrics import prec_rec_f1_acc_mcc
from data_processing import get_test_val_folds_train_data_loader
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

def compute_test_loss(model, criterion, data_loader, device, model_nm, extracted_layer, testFeatures):
    total_count = 0
    total_loss = 0.0
    all_comp_ids = []
    all_tar_ids = []
    all_labels = []
    predictions = []
    total_number_test = 0
    for i, data in enumerate(data_loader):

        if model_nm == "combined":
            comp_feature_vectors, target_feature_vectors, labels, compound_ids, target_ids = data
            comp_feature_vectors, target_feature_vectors, labels = Variable(comp_feature_vectors).to(
                device), Variable(
                target_feature_vectors).to(device), Variable(labels).to(device)
            all_comp_ids.extend(compound_ids)
            all_tar_ids.extend(target_ids)
            total_count += comp_feature_vectors.shape[0]
            if extracted_layer == "1":
                model.layer_combined.l1.register_forward_hook(get_activation('l1'))
                y_pred = model(comp_feature_vectors, target_feature_vectors).to(device)
                extractedFeatures = np.array(activation['l1'].cpu())
            elif extracted_layer == "2":
                model.layer_combined.l2.register_forward_hook(get_activation('l2'))
                y_pred = model(comp_feature_vectors, target_feature_vectors).to(device)
                extractedFeatures = np.array(activation['l2'].cpu())
        else:
            comp_feature_vectors, labels, compound_ids, target_ids = data
            comp_feature_vectors, labels = Variable(comp_feature_vectors).to(
                device), Variable(labels).to(device)
            all_comp_ids.extend(compound_ids)
            all_tar_ids.extend(target_ids)
            total_count += comp_feature_vectors.shape[0]
            if "conv1d" in model_nm:
                comp_feature_vectors = comp_feature_vectors[:, :, None]
            if extracted_layer == "1":
                model.l1.register_forward_hook(get_activation('l1'))
                y_pred = model(comp_feature_vectors).to(device)
                extractedFeatures = np.array(activation['l1'].cpu())
            elif extracted_layer == "2":
                model.l2.register_forward_hook(get_activation('l2'))
                y_pred = model(comp_feature_vectors).to(device)
                extractedFeatures = np.array(activation['l2'].cpu())
        if total_number_test == 0:
            testFeatures = extractedFeatures

        else:
            testFeatures = np.append(testFeatures, extractedFeatures, axis=0)

        total_number_test += len(labels)
        y_test_pred = torch.sigmoid(y_pred)
        y_pred_tag = torch.round(y_test_pred)
        loss_val = criterion(y_pred.squeeze(), labels)
        total_loss += float(loss_val.item())
        for item in labels:
            all_labels.append(float(item.item()))

        for item in y_pred_tag:
            predictions.append(float(item.item()))
    return total_loss, total_count, all_labels, predictions, all_comp_ids, all_tar_ids, testFeatures



activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def extract_features(target_dataset, source_dataset, comp_feature_list, tar_feature_list, comp_hidden_lst, tar_hidden_lst, learning_rate, batch_size,
                     model_nm, dropout, experiment_name, n_epoch, subset_flag, tl_flag, freeze_flag, freezing_layers,
                     extracted_layer, subset_size, input_path, setting):
    arguments = [str(argm) for argm in [target_dataset, source_dataset, comp_hidden_lst, tar_hidden_lst,  learning_rate, batch_size,
                                        experiment_name, model_nm, dropout, n_epoch, setting, subset_flag, tl_flag, freeze_flag,
                                        freezing_layers, extracted_layer, subset_size]]

    if model_nm != "combined":
        tar_feature_list = []
    str_arguments = "-".join(arguments)
    print("Arguments:", str_arguments)

    torch.manual_seed(101)
    np.random.seed(101)
    training_files_path = "{}/{}".format(input_path, "training_files")
    training_dataset_path = "{}/{}".format(training_files_path, target_dataset)
    trained_models_path = "{}/{}".format(input_path, "trained_models/")

    if model_nm == "combined":
        if tar_feature_list[0] == "AAC":
            tar_feature_size = 20
            tar_hidden_lst = [80, 20]
        elif tar_feature_list[0] == "TCGA-EMBEDDING":
            tar_feature_size = 50
            tar_hidden_lst = [200, 40]

        elif tar_feature_list[0] == "LEARNED-VEC":
            tar_feature_size = 64
            tar_hidden_lst = [64 * 4, 64]

        elif tar_feature_list[0] == "APAAC":
            tar_feature_size = 80
            tar_hidden_lst = [80 * 4, 80]

        elif tar_feature_list[0] == "PROTVEC":
            tar_feature_size = 100
            tar_hidden_lst = [100 * 4, 100]

        elif tar_feature_list[0] == "GENE2VEC":
            tar_feature_size = 200
            tar_hidden_lst = [200 * 4, 200]

        elif tar_feature_list[0] == "MUT2VEC":
            tar_feature_size = 300
            tar_hidden_lst = [300 * 4, 300]

        elif tar_feature_list[0] == "KSEP":
            tar_feature_size = 400
            tar_hidden_lst = [400 * 4, 400]

        elif tar_feature_list[0] == "CPC-PROT":
            tar_feature_size = 512
            tar_hidden_lst = [512 * 4, 512]

        elif tar_feature_list[0] == "BERT-PFAM":
            tar_feature_size = 768
            tar_hidden_lst = [768 * 4, 768]

        elif tar_feature_list[0] == "XLNET" or tar_feature_list[0] == "T5" or tar_feature_list[0] == "SEQVEC" \
                or tar_feature_list[0] == "BERT-BFD":
            tar_feature_size = 1024
            tar_hidden_lst = [1024 * 4, 1024]

        elif tar_feature_list[0] == "ESMB1":
            tar_feature_size = 1280
            tar_hidden_lst = [1280 * 4, 1280]

        elif tar_feature_list[0] == "ALBERT":
            tar_feature_size = 4096
            tar_hidden_lst = [4096 * 2, 2048]

        elif tar_feature_list[0] == "UNIREP":
            tar_feature_size = 5700
            tar_hidden_lst = [5700, 2850]

        elif tar_feature_list[0] == "PFAM":
            tar_feature_size = 6227
            tar_hidden_lst = [6227, 3113]

        elif tar_feature_list[0] == "BLAST" or tar_feature_list[0] == "HMMER":
            tar_feature_size = 20421
            tar_hidden_lst = [20421, 5105]

    loader_fold_dict, test_loader, external_data_loader = get_test_val_folds_train_data_loader(training_dataset_path,
                                                                                               comp_feature_list,
                                                                                               tar_feature_list,
                                                                                               batch_size, subset_size,
                                                                                               subset_flag, setting)
    num_of_folds = len(loader_fold_dict)
    folds = range(num_of_folds)
    for fold in folds:
        total_number_train = 0


        trainFeatures = np.empty((batch_size, comp_hidden_lst[int(extracted_layer) - 1]))
        valFeatures = np.empty((batch_size, comp_hidden_lst[int(extracted_layer) - 1]))
        testFeatures = np.empty((batch_size, comp_hidden_lst[int(extracted_layer) - 1]))
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
        if model_nm == "combined":
            model = get_model(model_nm, comp_feature_size, tar_feature_size, comp_hidden_lst, tar_hidden_lst, dropout)
        else:
            model = get_model(model_nm, 1024, 0, comp_hidden_lst, tar_hidden_lst, dropout)
        if tl_flag == 1:
            model.load_state_dict(torch.load("{}/{}/best_val-state_dict-fold-{}.pth".format(trained_models_path,
                                                                                             source_dataset, 0),
                                             map_location=torch.device(device)))
        model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        # freeze layers
        if freeze_flag == 1 and model_nm == "combined":
            if "1" in freezing_layers :
                model.layer_combined.l1.bias.requires_grad = False
                model.layer_combined.l1.weight.requires_grad = False
            if "2" in freezing_layers:
                model.layer_combined.l2.bias.requires_grad = False
                model.layer_combined.l2.weight.requires_grad = False
            if "3" in freezing_layers:
                model.layer_combined.l3.bias.requires_grad = False
                model.layer_combined.l3.weight.requires_grad = False
        if freeze_flag == 1 and model_nm != "combined":
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

                if model_nm == "combined":
                    comp_feature_vectors, target_feature_vectors, labels, compound_ids, target_ids = data
                    comp_feature_vectors, target_feature_vectors, labels = Variable(comp_feature_vectors).to(
                        device), Variable(
                        target_feature_vectors).to(device), Variable(labels).to(device)
                    y_pred = model(comp_feature_vectors, target_feature_vectors).to(device)
                else:
                    comp_feature_vectors, labels, compound_ids, target_ids = data
                    comp_feature_vectors, labels = Variable(comp_feature_vectors).to(
                        device), Variable(labels).to(device)
                    if "conv1d" in model_nm:
                        comp_feature_vectors = comp_feature_vectors[:, :, None]

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
                    if model_nm == "combined":
                        if extracted_layer == "1":
                            model.layer_combined.l1.register_forward_hook(get_activation('l1'))
                            y_pred = model(comp_feature_vectors, target_feature_vectors).to(device)
                            extractedFeatures = np.array(activation['l1'].cpu())
                        elif extracted_layer == "2":
                            model.layer_combined.l2.register_forward_hook(get_activation('l2'))
                            y_pred = model(comp_feature_vectors, target_feature_vectors).to(device)
                            extractedFeatures = np.array(activation['l2'].cpu())
                    else:
                        if "conv1d" in model_nm:
                            comp_feature_vectors = comp_feature_vectors[:, :, None]
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
                total_validation_loss, total_validation_count, validation_labels, validation_predictions, \
                all_val_comp_ids, all_val_tar_ids, valFeatures = compute_test_loss(
                    model, criterion, valid_loader, device, model_nm, extracted_layer, valFeatures)

                total_test_loss, total_test_count, test_labels, test_predictions, all_test_comp_ids, \
                all_test_tar_ids, testFeatures = compute_test_loss(
                    model, criterion, test_loader, device, model_nm, extracted_layer, testFeatures)

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

                np.savetxt(features_path + "/val.out", valFeatures, delimiter=',', fmt='%1.6f')
                np.savetxt(features_path + "/valClass.out", validation_labels, delimiter='\n', fmt='%1.6f')



                np.savetxt(features_path + "/test.out", testFeatures, delimiter=',', fmt='%1.6f')
                np.savetxt(features_path + "/testClass.out", test_labels, delimiter='\n', fmt='%1.6f')

