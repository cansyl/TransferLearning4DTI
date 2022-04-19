import torch
import torch.nn as nn
import torch.nn.functional as F

use_gpu = torch.cuda.is_available()
device = "cpu"
if use_gpu:
    device = "cuda"


class FC_2_Layer(torch.nn.Module):

    def __init__(self, number_of_inputs, neuron_l1, neuron_l2, drop_rate):
        super(FC_2_Layer, self).__init__()
        self.l1 = torch.nn.Linear(number_of_inputs, neuron_l1)
        self.bn1 = torch.nn.BatchNorm1d(num_features=neuron_l1)
        self.l2 = torch.nn.Linear(neuron_l1, neuron_l2)
        self.bn2 = torch.nn.BatchNorm1d(num_features=neuron_l2)
        self.layer_out = torch.nn.Linear(neuron_l2, 1)
        self.relu = torch.nn.ReLU()
        self.drop_rate = drop_rate

    def forward(self, x):
        out1 = F.dropout(self.relu(self.bn1(self.l1(x))), self.drop_rate)
        out2 = F.dropout(self.relu(self.bn2(self.l2(out1))), self.drop_rate)
        out3 = self.layer_out(out2)

        return out3


class FC_2_Layer_wout(torch.nn.Module):

    def __init__(self, number_of_inputs, neuron_l1, neuron_l2, drop_rate):
        super(FC_2_Layer_wout, self).__init__()
        self.l1 = torch.nn.Linear(number_of_inputs, neuron_l1)
        self.bn1 = torch.nn.BatchNorm1d(num_features=neuron_l1)
        self.l2 = torch.nn.Linear(neuron_l1, neuron_l2)
        self.bn2 = torch.nn.BatchNorm1d(num_features=neuron_l2)
        self.relu = torch.nn.ReLU()
        self.drop_rate = drop_rate

    def forward(self, x):
        out1 = F.dropout(self.relu(self.bn1(self.l1(x))), self.drop_rate)
        out2 = F.dropout(self.relu(self.bn2(self.l2(out1))), self.drop_rate)

        return out2


class FC_3_Layer(torch.nn.Module):

    def __init__(self, number_of_inputs, neuron_l1, neuron_l2, neuron_l3, drop_rate):
        super(FC_3_Layer, self).__init__()
        self.l1 = torch.nn.Linear(number_of_inputs, neuron_l1)
        self.bn1 = torch.nn.BatchNorm1d(num_features=neuron_l1)

        self.l2 = torch.nn.Linear(neuron_l1, neuron_l2)
        self.bn2 = torch.nn.BatchNorm1d(num_features=neuron_l2)

        self.l3 = torch.nn.Linear(neuron_l2, neuron_l3)
        self.bn3 = torch.nn.BatchNorm1d(num_features=neuron_l3)

        self.relu = torch.nn.ReLU()
        self.drop_rate = drop_rate

    def forward(self, x):
        out1 = F.dropout(self.relu(self.bn1(self.l1(x))), self.drop_rate)
        out2 = F.dropout(self.relu(self.bn2(self.l2(out1))), self.drop_rate)
        out3 = F.dropout(self.relu(self.bn3(self.l3(out2))), self.drop_rate)

        return out3


class FC_3_Layer_wout(torch.nn.Module):

    def __init__(self, number_of_inputs, neuron_l1, neuron_l2, neuron_l3, drop_rate):
        super(FC_3_Layer, self).__init__()
        self.l1 = torch.nn.Linear(number_of_inputs, neuron_l1)
        self.bn1 = torch.nn.BatchNorm1d(num_features=neuron_l1)

        self.l2 = torch.nn.Linear(neuron_l1, neuron_l2)
        self.bn2 = torch.nn.BatchNorm1d(num_features=neuron_l2)

        self.l3 = torch.nn.Linear(neuron_l2, neuron_l3)
        self.bn3 = torch.nn.BatchNorm1d(num_features=neuron_l3)

        self.layer_out = torch.nn.Linear(neuron_l3, 1)

        self.relu = torch.nn.ReLU()
        self.drop_rate = drop_rate

    def forward(self, x):
        out1 = F.dropout(self.relu(self.bn1(self.l1(x))), self.drop_rate)
        out2 = F.dropout(self.relu(self.bn2(self.l2(out1))), self.drop_rate)
        out3 = F.dropout(self.relu(self.bn3(self.l3(out2))), self.drop_rate)
        out4 = self.layer_out(out3)
        return out4


class FC_4_Layer(torch.nn.Module):

    def __init__(self, number_of_inputs, neuron_l1, neuron_l2, neuron_l3, neuron_l4, drop_rate):
        super(FC_4_Layer, self).__init__()
        self.l1 = torch.nn.Linear(number_of_inputs, neuron_l1)
        self.bn1 = torch.nn.BatchNorm1d(num_features=neuron_l1)

        self.l2 = torch.nn.Linear(neuron_l1, neuron_l2)
        self.bn2 = torch.nn.BatchNorm1d(num_features=neuron_l2)

        self.l3 = torch.nn.Linear(neuron_l2, neuron_l3)
        self.bn3 = torch.nn.BatchNorm1d(num_features=neuron_l3)

        self.l4 = torch.nn.Linear(neuron_l3, neuron_l4)
        self.bn4 = torch.nn.BatchNorm1d(num_features=neuron_l4)

        self.layer_out = torch.nn.Linear(neuron_l4, 1)

        self.relu = torch.nn.ReLU()
        self.drop_rate = drop_rate

    def forward(self, x):
        out1 = F.dropout(self.relu(self.bn1(self.l1(x))), self.drop_rate)
        out2 = F.dropout(self.relu(self.bn2(self.l2(out1))), self.drop_rate)
        out3 = F.dropout(self.relu(self.bn3(self.l3(out2))), self.drop_rate)
        out4 = F.dropout(self.relu(self.bn4(self.l4(out3))), self.drop_rate)
        out5 = self.layer_out(out4)
        return out5


class Conv1d_2_layer(nn.Module):
    def __init__(self, input_size, l1, l2, drop_rate):
        super().__init__()

        # Extract features, 1D conv layers
        self.features = nn.Sequential(
            nn.Conv1d(input_size, l1, 1),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Conv1d(l1, l2, 1),
            nn.ReLU(),
        )
        # Classify output, fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(l2, l2),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(l2, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 1024)
        out = self.classifier(x)

        return out


class SoftOrdering1DCNN(nn.Module):

    def __init__(self, input_dim, output_dim, sign_size=32, cha_input=16, cha_hidden=32,
                 K=2, dropout_input=0.2, dropout_hidden=0.2, dropout_output=0.2):
        super().__init__()

        hidden_size = sign_size * cha_input
        sign_size1 = sign_size
        sign_size2 = sign_size // 2
        output_size = (sign_size // 4) * cha_hidden

        self.hidden_size = hidden_size
        self.cha_input = cha_input
        self.cha_hidden = cha_hidden
        self.K = K
        self.sign_size1 = sign_size1
        self.sign_size2 = sign_size2
        self.output_size = output_size
        self.dropout_input = dropout_input
        self.dropout_hidden = dropout_hidden
        self.dropout_output = dropout_output

        self.batch_norm1 = nn.BatchNorm1d(input_dim)
        self.dropout1 = nn.Dropout(dropout_input)
        dense1 = nn.Linear(input_dim, hidden_size, bias=False)
        self.dense1 = nn.utils.weight_norm(dense1)

        # 1st conv layer
        self.batch_norm_c1 = nn.BatchNorm1d(cha_input)
        conv1 = conv1 = nn.Conv1d(
            cha_input,
            cha_input * K,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=cha_input,
            bias=False)
        self.conv1 = nn.utils.weight_norm(conv1, dim=None)

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size=sign_size2)

        # 2nd conv layer
        self.batch_norm_c2 = nn.BatchNorm1d(cha_input * K)
        self.dropout_c2 = nn.Dropout(dropout_hidden)
        conv2 = nn.Conv1d(
            cha_input * K,
            cha_hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.conv2 = nn.utils.weight_norm(conv2, dim=None)

        # 3rd conv layer
        self.batch_norm_c3 = nn.BatchNorm1d(cha_hidden)
        self.dropout_c3 = nn.Dropout(dropout_hidden)
        conv3 = nn.Conv1d(
            cha_hidden,
            cha_hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.conv3 = nn.utils.weight_norm(conv3, dim=None)

        # 4th conv layer
        self.batch_norm_c4 = nn.BatchNorm1d(cha_hidden)
        conv4 = nn.Conv1d(
            cha_hidden,
            cha_hidden,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=cha_hidden,
            bias=False)
        self.conv4 = nn.utils.weight_norm(conv4, dim=None)

        self.avg_po_c4 = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.batch_norm2 = nn.BatchNorm1d(output_size)
        self.dropout2 = nn.Dropout(dropout_output)
        dense2 = nn.Linear(output_size, output_dim, bias=False)
        self.dense2 = nn.utils.weight_norm(dense2)

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = nn.functional.celu(self.dense1(x))

        x = x.reshape(x.shape[0], self.cha_input, self.sign_size1)

        x = self.batch_norm_c1(x)
        x = nn.functional.relu(self.conv1(x))

        x = self.ave_po_c1(x)

        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = nn.functional.relu(self.conv2(x))
        x_s = x

        x = self.batch_norm_c3(x)
        x = self.dropout_c3(x)
        x = nn.functional.relu(self.conv3(x))

        x = self.batch_norm_c4(x)
        x = self.conv4(x)
        x = x + x_s
        x = nn.functional.relu(x)

        x = self.avg_po_c4(x)

        x = self.flt(x)

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.dense2(x)

        return x

class Model_DNN(nn.Module):
    def __init__(self, num_features, num_targets):
        super(Model_DNN, self).__init__()
        self.hidden_size = [1500, 1250, 1000, 750]
        self.dropout_value = [0.5, 0.35, 0.3, 0.25]

        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.Linear(num_features, self.hidden_size[0])

        self.batch_norm2 = nn.BatchNorm1d(self.hidden_size[0])
        self.dropout2 = nn.Dropout(self.dropout_value[0])
        self.dense2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])

        self.batch_norm3 = nn.BatchNorm1d(self.hidden_size[1])
        self.dropout3 = nn.Dropout(self.dropout_value[1])
        self.dense3 = nn.Linear(self.hidden_size[1], self.hidden_size[2])

        self.batch_norm4 = nn.BatchNorm1d(self.hidden_size[2])
        self.dropout4 = nn.Dropout(self.dropout_value[2])
        self.dense4 = nn.Linear(self.hidden_size[2], self.hidden_size[3])

        self.batch_norm5 = nn.BatchNorm1d(self.hidden_size[3])
        self.dropout5 = nn.Dropout(self.dropout_value[3])
        self.dense5 = nn.utils.weight_norm(nn.Linear(self.hidden_size[3], num_targets))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = F.leaky_relu(self.dense3(x))

        x = self.batch_norm4(x)
        x = self.dropout4(x)
        x = F.leaky_relu(self.dense4(x))

        x = self.batch_norm5(x)
        x = self.dropout5(x)
        x = self.dense5(x)
        return x

class Model_1DCNN(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):
        super(Model_1DCNN, self).__init__()
        cha_1 = 128
        cha_2 = 128
        cha_3 = 128

        cha_1_reshape = int(hidden_size / cha_1)
        cha_po_1 = int(hidden_size / cha_1 / 2)
        cha_po_2 = int(hidden_size / cha_1 / 2 / 2) * cha_3

        self.cha_1 = cha_1
        self.cha_2 = cha_2
        self.cha_3 = cha_3
        self.cha_1_reshape = cha_1_reshape
        self.cha_po_1 = cha_po_1
        self.cha_po_2 = cha_po_2

        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.1)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm_c1 = nn.BatchNorm1d(cha_1)
        self.dropout_c1 = nn.Dropout(0.1)
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(cha_1, cha_2, kernel_size=5, stride=1, padding=2, bias=False),
                                          dim=None)

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size=cha_po_1)

        self.batch_norm_c2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2 = nn.Dropout(0.1)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(cha_2, cha_2, kernel_size=3, stride=1, padding=1, bias=True),
                                          dim=None)

        self.batch_norm_c2_1 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_1 = nn.Dropout(0.3)
        self.conv2_1 = nn.utils.weight_norm(nn.Conv1d(cha_2, cha_2, kernel_size=3, stride=1, padding=1, bias=True),
                                            dim=None)

        self.batch_norm_c2_2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_2 = nn.Dropout(0.2)
        self.conv2_2 = nn.utils.weight_norm(nn.Conv1d(cha_2, cha_3, kernel_size=5, stride=1, padding=2, bias=True),
                                            dim=None)

        self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.batch_norm3 = nn.BatchNorm1d(cha_po_2)
        self.dropout3 = nn.Dropout(0.2)
        self.dense3 = nn.utils.weight_norm(nn.Linear(cha_po_2, num_targets))

    def forward(self, x):
        # x = self.batch_norm1(x)
        # x = self.dropout1(x)
        x = F.celu(self.dense1(x), alpha=0.06)
        # print(x.shape)
        x = x.reshape(x.shape[0], self.cha_1,
                      self.cha_1_reshape)
        # print(x.shape)

        x = self.batch_norm_c1(x)
        x = self.dropout_c1(x)
        x = F.relu(self.conv1(x))

        x = self.ave_po_c1(x)

        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x_s = x

        x = self.batch_norm_c2_1(x)
        x = self.dropout_c2_1(x)
        x = F.relu(self.conv2_1(x))

        x = self.batch_norm_c2_2(x)
        x = self.dropout_c2_2(x)
        x = F.relu(self.conv2_2(x))
        x = x * x_s

        x = self.max_po_c2(x)
        # print(x.shape)

        x = self.flt(x)
        # print(x.shape)

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)
        # print(x.shape)

        return x

class ConvNet1D(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(n_features, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(10))
        self.layer2 = nn.Flatten()
        self.layer3 = nn.Sequential(
            nn.Linear(768,100),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Linear(100,1))

    def forward(self, x):
        print(x.shape)
        out = self.layer1(x)
        print(out.shape)
        out = self.layer2(out)
        print(out.shape)

        out = self.layer3(out)
        print(out.shape)

        out = self.layer4(out)
        print(out.shape)

        return out

class CompFCNNTarFCNNModuleInception(torch.nn.Module):
    def __init__(self, number_of_comp_features, number_of_tar_features, comp_l1, comp_l2, tar_l1, tar_l2, drop_prob):
        super(CompFCNNTarFCNNModuleInception, self).__init__()
        print(number_of_comp_features, number_of_tar_features, comp_l1, comp_l2, tar_l1, tar_l2, drop_prob)
        # print(tar_feature_list)
        self.layer_comp = FC_2_Layer_wout(number_of_comp_features, comp_l1, comp_l2, drop_prob)
        self.layer_tar = FC_2_Layer_wout(number_of_tar_features, tar_l1, tar_l2, drop_prob)
        self.layer_combined = FC_2_Layer_wout(comp_l2 + tar_l2, comp_l2 + tar_l2, comp_l2 + tar_l2, drop_prob)
        self.output = None

        self.dropout = torch.nn.Dropout(drop_prob)
        self.output = torch.nn.Linear(comp_l2 + tar_l2, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x_comp, x_tar):
        out_comp = self.layer_comp.forward(x_comp)
        out_tar = self.layer_tar.forward(x_tar)
        combined_layer = torch.cat((out_comp, out_tar), 1)
        out_combined = self.layer_combined.forward(combined_layer)
        pred = self.output.forward(out_combined)

        return pred


def get_model(model_name, num_of_comp_features, num_of_tar_features, comp_hidden_lst, tar_hidden_lst, dropout):
    model = None
    hidden_size = 4096

    if model_name == "fc_2_layer":
        model = FC_2_Layer(num_of_comp_features, comp_hidden_lst[0], comp_hidden_lst[1], dropout)

    elif model_name == "fc_3_layer":
        model = FC_3_Layer(num_of_comp_features, comp_hidden_lst[0], comp_hidden_lst[1], comp_hidden_lst[2], dropout)

    elif model_name == "fc_4_layer":
        model = FC_4_Layer(num_of_comp_features, comp_hidden_lst[0], comp_hidden_lst[1], comp_hidden_lst[2],
                           comp_hidden_lst[3], dropout)

    elif model_name == "DNN":
        model = Model_DNN(num_of_comp_features, 1)

    elif model_name == "1DCNN":
        model = Model_1DCNN(num_of_comp_features, 1, hidden_size)

    elif model_name == "ConvNet1D":
        model = ConvNet1D(num_of_comp_features)

    elif model_name == "combined":
        model = CompFCNNTarFCNNModuleInception(num_of_comp_features, num_of_tar_features, comp_hidden_lst[0],
                                               comp_hidden_lst[1], tar_hidden_lst[0], tar_hidden_lst[1], dropout)

    elif model_name == "conv1d_2_layer":
        model = Conv1d_2_layer(num_of_comp_features, comp_hidden_lst[0], comp_hidden_lst[1], dropout)

    return model
