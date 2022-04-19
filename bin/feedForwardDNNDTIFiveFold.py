import os
import warnings
from builtins import len

from models import get_model
from evaluation_metrics import prec_rec_f1_acc_mcc, get_list_of_scores
from data_processing import get_test_val_folds_train_data_loader, get_train_test_train_data_loader, get_train_data_loader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# import wandb
# wandb.login(key="4d794129c1cc9b661a9e19ec8e803253fa85bd1c")
# wandb.init(project="nuclearreceptor-sub", entity="transferlearning")
warnings.filterwarnings(action='ignore')


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def compute_test_loss(model, criterion, data_loader, device, model_nm):
    total_count = 0
    total_loss = 0.0
    all_comp_ids = []
    all_tar_ids = []
    all_labels = []
    predictions = []
    for i, data in enumerate(data_loader):

        if model_nm == "combined":
            comp_feature_vectors, target_feature_vectors, labels, compound_ids, target_ids = data
            comp_feature_vectors, target_feature_vectors, labels = Variable(comp_feature_vectors).to(
                device), Variable(
                target_feature_vectors).to(device), Variable(labels).to(device)
            all_comp_ids.extend(compound_ids)
            all_tar_ids.extend(target_ids)
            total_count += comp_feature_vectors.shape[0]
            y_pred = model(comp_feature_vectors, target_feature_vectors).to(device)
        else:
            comp_feature_vectors, labels, compound_ids, target_ids = data
            comp_feature_vectors, labels = Variable(comp_feature_vectors).to(
                device), Variable(labels).to(device)
            all_comp_ids.extend(compound_ids)
            all_tar_ids.extend(target_ids)
            total_count += comp_feature_vectors.shape[0]
            if "conv1d" in model_nm:
                comp_feature_vectors = comp_feature_vectors[:, :, None]
            y_pred = model(comp_feature_vectors).to(device)
        y_test_pred = torch.sigmoid(y_pred)
        y_pred_tag = torch.round(y_test_pred)
        loss_val = criterion(y_pred.squeeze(), labels)
        total_loss += float(loss_val.item())
        for item in labels:
            all_labels.append(float(item.item()))

        for item in y_pred_tag:
            predictions.append(float(item.item()))
    return total_loss, total_count, all_labels, predictions, all_comp_ids, all_tar_ids


def get_external_test_results(model, data_loader, device, model_nm):
    total_count = 0
    all_comp_ids = []
    predictions = []
    results_dict = {}

    for i, data in enumerate(data_loader):
        comp_feature_vectors, compound_ids = data
        comp_feature_vectors = Variable(comp_feature_vectors).to(
            device)
        all_comp_ids.extend(compound_ids)
        total_count += comp_feature_vectors.shape[0]
        if "conv1d" in model_nm:
            comp_feature_vectors = comp_feature_vectors[:, :, None]
        y_pred = model(comp_feature_vectors).to(device)
        y_test_pred = torch.sigmoid(y_pred)

        for i in range(len(compound_ids)):
            results_dict[compound_ids[i]] = y_test_pred[i].item()
        for item in y_test_pred:
            predictions.append(float(item.item()))
    return results_dict


def save_best_model_predictions(trained_models_path, experiment_name, model, fold):
    if not os.path.exists(os.path.join(trained_models_path, experiment_name)):
        os.makedirs(os.path.join(trained_models_path, experiment_name))



    torch.save(model.state_dict(),
               "{}/{}/best_val-state_dict-fold-{}.pth".format(trained_models_path, experiment_name, fold))



def five_fold_training(target_dataset, source_dataset, comp_feature_list, tar_feature_list, comp_hidden_lst, tar_hidden_lst, learning_rate, batch_size,
                       model_nm, dropout, experiment_name, n_epoch, subset_flag, tl_flag, freeze_flag, freezing_layers,
                       subset_size, input_path, output_path, setting):
    arguments = [str(argm) for argm in [target_dataset, source_dataset, comp_feature_list, tar_feature_list, comp_hidden_lst, tar_hidden_lst, learning_rate, batch_size,
                                        experiment_name, model_nm, dropout, n_epoch, setting,
                                        subset_flag, tl_flag, freeze_flag, freezing_layers, subset_size]]

    str_arguments = "-".join(arguments)
    print("Arguments:", str_arguments)

    torch.manual_seed(101)
    np.random.seed(101)
    if model_nm != "combined":
        tar_feature_list = []

    training_files_path = "{}/{}".format(input_path, "training_files")
    training_dataset_path = "{}/{}".format(training_files_path, target_dataset)
    result_files_path = "{}/{}".format(output_path, "result_files/")
    trained_models_path = "{}/{}".format(input_path, "trained_models/")




    if subset_flag == 0:
        exp_path = os.path.join(result_files_path, target_dataset)
    else:
        exp_path = os.path.join(result_files_path, target_dataset + "/dataSubset" + str(subset_size))

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    if not os.path.exists(exp_path + "/scratch"):
        os.makedirs(exp_path + "/scratch")
    if not os.path.exists(exp_path + "/freeze"):
        os.makedirs(exp_path + "/freeze")
    if not os.path.exists(exp_path + "/fine-tuned"):
        os.makedirs(exp_path + "/fine-tuned")

    if tl_flag == 0:
        tl = "scratch"
    else:
        tl = "fine-tuned"
    if subset_flag == 0 or (subset_flag == 1 and tl_flag == 0):
        best_val_test_result_fl = open("{}/scratch/{}_perf_results-{}.txt".format(exp_path, tl, str_arguments), "w")
    elif subset_flag == 1:
        if freeze_flag == 1:
            best_val_test_result_fl = open("{}/freeze/sub_{}_freeze_{}_perf_results-{}.txt".format(
                exp_path, tl, freezing_layers, str_arguments), "w")
        else:
            best_val_test_result_fl = open(
                "{}/fine-tuned/sub_{}_perf_results-{}.txt".format(exp_path, tl, str_arguments), "w")
    if model_nm == "combined":
        if tar_feature_list[0] == "AAC":
            tar_feature_size = 20
            tar_hidden_lst = [80,20]
        elif tar_feature_list[0] == "TCGA-EMBEDDING":
            tar_feature_size = 50
            tar_hidden_lst = [200,40]

        elif tar_feature_list[0] == "LEARNED-VEC":
            tar_feature_size = 64
            tar_hidden_lst = [64*4,64]

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

        elif tar_feature_list[0] == "XLNET" or tar_feature_list[0] == "T5" or tar_feature_list[0] == "SEQVEC"\
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
            tar_hidden_lst = [4096 * 2, 2048]

    loader_fold_dict, test_loader, external_data_loader = get_test_val_folds_train_data_loader(training_dataset_path,
                                                                                               comp_feature_list,
                                                                                               tar_feature_list,
                                                                                               batch_size, subset_size,
                                                                                               subset_flag, setting)
    num_of_folds = len(loader_fold_dict)
    folds = range(num_of_folds)

    average_validation_mcc, average_test_mcc = 0, 0
    for fold in folds:

        print("FOLD:", fold + 1)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        train_loader, valid_loader = loader_fold_dict[fold]
        if comp_feature_list[0] == "ecfp4":
            comp_feature_size = 1024
        if model_nm == "combined":
            model = get_model(model_nm, comp_feature_size, tar_feature_size, comp_hidden_lst, tar_hidden_lst, dropout)

        else:
            model = get_model(model_nm, comp_feature_size, 0, comp_hidden_lst, tar_hidden_lst, dropout)
        if tl_flag == 1:
            model.load_state_dict(
                torch.load("{}/{}/best_val-state_dict-fold-{}.pth".format(trained_models_path, source_dataset, 0),
                           map_location=torch.device(device)))
        model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        # freeze layers
        if freeze_flag == 1 and model_nm == "combined":
            if "1" in freezing_layers:
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
        best_val_score_epoch, best_test_score_epoch = 0, 0
        best_val_mcc_score, best_test_mcc_score = -10000.0, -10000.0
        best_val_test_performance_dict = dict()
        best_val_test_performance_dict["MCC"] = 0.0
        model.train()
        for epoch in range(1, n_epoch + 1):
            total_training_loss, total_validation_loss, total_test_loss = 0.0, 0.0, 0.0
            epoch_loss = 0
            epoch_acc = 0

            for i, data in enumerate(train_loader):
                # clear gradient DO NOT forget you fool!
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
                    if "conv1d" in model_nm or "ConvNet1D" in model_nm:
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

            model.eval()
            with torch.no_grad():  # torch.set_grad_enabled(False):
                # print("Training:", model.training)
                total_val_loss, total_val_count, val_labels, val_predictions, all_val_comp_ids, all_val_tar_ids = \
                    compute_test_loss(model, criterion, valid_loader, device, model_nm)
                # print("Epoch {} validation loss:".format(epoch), total_validation_loss)

                total_test_loss, total_test_count, test_labels, test_predictions, all_test_comp_ids, all_test_tar_ids =\
                    compute_test_loss(model, criterion, test_loader, device, model_nm)
                # print("Epoch {} test loss:".format(epoch), total_test_loss)


            # wandb.log({"val_loss": val_loss})
            val_perf_dict = dict()
            val_perf_dict["MCC"] = 0.0

            val_perf_dict = prec_rec_f1_acc_mcc(val_labels, val_predictions)
            # wandb.log({"val_mcc": val_perf_dict["MCC"]})

            test_perf_dict = dict()
            test_perf_dict["MCC"] = 0.0

            test_perf_dict = prec_rec_f1_acc_mcc(test_labels, test_predictions)
            # wandb.log({"test_mcc": test_perf_dict["MCC"]})
            print(f'Epoch {epoch + 0:03}: | Loss: {total_training_loss:.5f}  | Val_loss: {total_val_loss:.5f} '
                  f'| Acc: {epoch_acc / len(train_loader):.3f} | Val_MCC: {val_perf_dict["MCC"]:.4f} '
                  f'| Test_MCC: {test_perf_dict["MCC"]:.4f}')

            if val_perf_dict["MCC"] > best_val_mcc_score:
                best_val_mcc_score = val_perf_dict["MCC"]
                best_val_performance_dict = val_perf_dict
                best_val_score_epoch = epoch
                if subset_flag == 0:
                    save_best_model_predictions(trained_models_path, target_dataset, model, fold)

            if test_perf_dict["MCC"] > best_test_mcc_score:
                best_test_mcc_score = test_perf_dict["MCC"]
                best_test_performance_dict = test_perf_dict
                best_test_score_epoch = epoch

            if epoch == n_epoch:
                print(best_val_performance_dict, "in epoch:", best_val_score_epoch)
                print(best_test_performance_dict, "in epoch:", best_test_score_epoch)
                average_validation_mcc += best_val_performance_dict["MCC"]
                average_test_mcc += best_test_performance_dict["MCC"]

                score_list = get_list_of_scores()
                best_val_test_result_fl.write("FOLD : {}\n".format(fold + 1))
                for scr in score_list:
                    best_val_test_result_fl.write("Val {}:\t{}\n".format(scr, best_val_performance_dict[scr]))
                for scr in score_list:
                    best_val_test_result_fl.write("Test {}:\t{}\n".format(scr, best_test_performance_dict[scr]))
                if fold == 4:
                    average_validation_mcc /= 5
                    average_test_mcc /= 5
                    print("average best validation mcc:", average_validation_mcc)
                    print("average best test mcc:", average_test_mcc)
                    best_val_test_result_fl.write("Val avg mcc:\t{}\n".format(average_validation_mcc))
                    best_val_test_result_fl.write("Test avg mcc:\t{}\n".format(average_test_mcc))
    best_val_test_result_fl.close()


def training_test(target_dataset, source_dataset, comp_feature_list, tar_feature_list, comp_hidden_lst, tar_hidden_lst, learning_rate, batch_size,
                       model_nm, dropout, experiment_name, n_epoch, subset_flag, tl_flag, freeze_flag, freezing_layers,
                       subset_size, input_path, output_path, setting):
    arguments = [str(argm) for argm in [target_dataset, source_dataset, comp_feature_list, tar_feature_list, comp_hidden_lst, tar_hidden_lst, learning_rate, batch_size,
                                        experiment_name, model_nm, dropout, n_epoch, setting,
                                        subset_flag, tl_flag, freeze_flag, freezing_layers, subset_size]]

    str_arguments = "-".join(arguments)
    print("Arguments:", str_arguments)

    torch.manual_seed(101)
    np.random.seed(101)
    if model_nm != "combined":
        tar_feature_list = []

    training_files_path = "{}/{}".format(input_path, "training_files")
    training_dataset_path = "{}/{}".format(training_files_path, target_dataset)
    result_files_path = "{}/{}".format(output_path, "result_files/")
    trained_models_path = "{}/{}".format(input_path, "trained_models/")

    train_loader, test_loader, external_data_loader = get_train_test_train_data_loader(training_dataset_path,
                                                                                               comp_feature_list,
                                                                                               tar_feature_list,
                                                                                               batch_size, subset_size,
                                                                                               subset_flag, setting)
    if model_nm == "combined":
        if tar_feature_list[0] == "AAC":
            tar_feature_size = 20
            tar_hidden_lst = [80,20]
        elif tar_feature_list[0] == "TCGA-EMBEDDING":
            tar_feature_size = 50
            tar_hidden_lst = [200,40]

        elif tar_feature_list[0] == "LEARNED-VEC":
            tar_feature_size = 64
            tar_hidden_lst = [64*4,64]

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

        elif tar_feature_list[0] == "XLNET" or tar_feature_list[0] == "T5" or tar_feature_list[0] == "SEQVEC"\
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



    if subset_flag == 0:
        exp_path = os.path.join(result_files_path, target_dataset)
    else:
        exp_path = os.path.join(result_files_path, target_dataset + "/dataSubset" + str(subset_size))

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    if not os.path.exists(exp_path + "/scratch"):
        os.makedirs(exp_path + "/scratch")
    if not os.path.exists(exp_path + "/freeze"):
        os.makedirs(exp_path + "/freeze")
    if not os.path.exists(exp_path + "/fine-tuned"):
        os.makedirs(exp_path + "/fine-tuned")

    if tl_flag == 0:
        tl = "scratch"
    else:
        tl = "fine-tuned"
    if subset_flag == 0 or (subset_flag == 1 and tl_flag == 0):
        best_val_test_result_fl = open("{}/scratch/{}_perf_results-{}.txt".format(exp_path, tl, str_arguments), "w")
    elif subset_flag == 1:
        if freeze_flag == 1:
            best_val_test_result_fl = open("{}/freeze/sub_{}_freeze_{}_perf_results-{}.txt".format(
                exp_path, tl, freezing_layers, str_arguments), "w")
        else:
            best_val_test_result_fl = open(
                "{}/fine-tuned/sub_{}_perf_results-{}.txt".format(exp_path, tl, str_arguments), "w")


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if comp_feature_list[0] == "ecfp4":
        comp_feature_size = 1024
    if model_nm == "combined":
        model = get_model(model_nm, comp_feature_size, tar_feature_size, comp_hidden_lst, tar_hidden_lst, dropout)

    else:
        model = get_model(model_nm, comp_feature_size, 0, comp_hidden_lst, tar_hidden_lst, dropout)
    if tl_flag == 1:
        model.load_state_dict(
            torch.load("{}/{}/best_val-state_dict-fold-{}.pth".format(trained_models_path, source_dataset, 0),
                       map_location=torch.device(device)))
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    # freeze layers
    if freeze_flag == 1 and model_nm == "combined":
        if "1" in freezing_layers:
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
    best_val_score_epoch, best_test_score_epoch = 0, 0
    best_val_mcc_score, best_test_mcc_score = -10000.0, -10000.0
    best_val_test_performance_dict = dict()
    best_val_test_performance_dict["MCC"] = 0.0
    model.train()
    for epoch in range(1, n_epoch + 1):
        total_training_loss, total_validation_loss, total_test_loss = 0.0, 0.0, 0.0
        epoch_loss = 0
        epoch_acc = 0

        for i, data in enumerate(train_loader):
            # clear gradient DO NOT forget you fool!
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

        model.eval()
        with torch.no_grad():  # torch.set_grad_enabled(False):
            total_test_loss, total_test_count, test_labels, test_predictions, all_test_comp_ids, all_test_tar_ids =\
                compute_test_loss(model, criterion, test_loader, device, model_nm)
            # print("Epoch {} test loss:".format(epoch), total_test_loss)


        test_perf_dict = dict()
        test_perf_dict["MCC"] = 0.0

        test_perf_dict = prec_rec_f1_acc_mcc(test_labels, test_predictions)
        # wandb.log({"test_mcc": test_perf_dict["MCC"]})
        print(f'Epoch {epoch + 0:03}: | Loss: {total_training_loss:.5f}  '
              f'| Acc: {epoch_acc / len(train_loader):.3f} '
              f'| Test_MCC: {test_perf_dict["MCC"]:.4f}')


        if test_perf_dict["MCC"] > best_test_mcc_score:
            best_test_mcc_score = test_perf_dict["MCC"]
            best_test_performance_dict = test_perf_dict
            best_test_score_epoch = epoch

        if epoch == n_epoch:
            print(best_test_performance_dict, "in epoch:", best_test_score_epoch)
            score_list = get_list_of_scores()
            for scr in score_list:
                best_val_test_result_fl.write("Test {}:\t{}\n".format(scr, best_test_performance_dict[scr]))

    best_val_test_result_fl.close()

def training(target_dataset, source_dataset, comp_feature_list, tar_feature_list, comp_hidden_lst,
                  tar_hidden_lst, learning_rate, batch_size,
                  model_nm, dropout, experiment_name, n_epoch, subset_flag, tl_flag, freeze_flag, freezing_layers,
                  subset_size, input_path, output_path, external_file):
    arguments = [str(argm) for argm in
                 [target_dataset, source_dataset, comp_feature_list, tar_feature_list, comp_hidden_lst,
                  tar_hidden_lst, learning_rate, batch_size,
                  experiment_name, model_nm, dropout, n_epoch, external_file,
                  subset_flag, tl_flag, freeze_flag, freezing_layers, subset_size]]

    str_arguments = "-".join(arguments)
    print("Arguments:", str_arguments)

    torch.manual_seed(101)
    np.random.seed(101)
    if model_nm != "combined":
        tar_feature_list = []

    training_files_path = "{}/{}".format(input_path, "training_files")
    training_dataset_path = "{}/{}".format(training_files_path, target_dataset)
    trained_models_path = "{}/{}".format(input_path, "trained_models/")

    train_loader, external_test_loader = get_train_data_loader(training_dataset_path, comp_feature_list,
                                                                                       tar_feature_list,
                                                                                       batch_size, subset_size,
                                                                                       subset_flag, external_file)
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if comp_feature_list[0] == "ecfp4":
        comp_feature_size = 1024
    if model_nm == "combined":
        model = get_model(model_nm, comp_feature_size, tar_feature_size, comp_hidden_lst, tar_hidden_lst, dropout)

    else:
        model = get_model(model_nm, comp_feature_size, 0, comp_hidden_lst, tar_hidden_lst, dropout)
    if tl_flag == 1:
        model.load_state_dict(
            torch.load("{}/{}/best_val-state_dict.pth".format(trained_models_path, source_dataset),
                       map_location=torch.device(device)))
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    # freeze layers
    if freeze_flag == 1 and model_nm == "combined":
        if "1" in freezing_layers:
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
    best_val_score_epoch, best_test_score_epoch = 0, 0
    best_val_mcc_score, best_test_mcc_score = -10000.0, -10000.0
    best_val_test_performance_dict = dict()
    best_val_test_performance_dict["MCC"] = 0.0
    model.train()
    for epoch in range(1, n_epoch + 1):
        total_training_loss, total_validation_loss, total_test_loss = 0.0, 0.0, 0.0
        epoch_loss = 0
        epoch_acc = 0

        for i, data in enumerate(train_loader):
            # clear gradient DO NOT forget you fool!
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

        print(f'Epoch {epoch + 0:03}: | Loss: {total_training_loss:.5f}  '
              f'| Acc: {epoch_acc / len(train_loader):.3f} ')

        model.eval()
        with torch.no_grad():  # torch.set_grad_enabled(False):
            results_dict = get_external_test_results(model, external_test_loader, device, model_nm)
        # wandb.log({"test_mcc": test_perf_dict["MCC"]})


        if epoch == n_epoch:
            save_best_model_predictions(trained_models_path, target_dataset, model, 5)
            # for pred in test_predictions:
            #     print()
            print(results_dict)


