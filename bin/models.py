import torch
import torch.nn.functional as f

use_gpu = torch.cuda.is_available()
device = "cpu"
if use_gpu:
    device = "cuda"


class FC_2_Layer(torch.nn.Module):

    def __init__(self, number_of_inputs, neuron_l1, neuron_l2, num_class, drop_rate):
        super(FC_2_Layer, self).__init__()
        self.l1 = torch.nn.Linear(number_of_inputs, neuron_l1)
        self.bn1 = torch.nn.BatchNorm1d(num_features=neuron_l1)
        self.l2 = torch.nn.Linear(neuron_l1, neuron_l2)
        self.bn2 = torch.nn.BatchNorm1d(num_features=neuron_l2)
        if num_class == 2:
            self.layer_out = torch.nn.Linear(neuron_l2, 1)
        else:
            self.layer_out = torch.nn.Linear(neuron_l2, num_class)
        self.relu = torch.nn.ReLU()
        self.drop_rate = drop_rate

    def forward(self, x):
        out1 = f.dropout(self.relu(self.bn1(self.l1(x))), self.drop_rate)
        out2 = f.dropout(self.relu(self.bn2(self.l2(out1))), self.drop_rate)
        out3 = self.layer_out(out2)

        return out3


class FC_3_Layer(torch.nn.Module):

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
        out1 = f.dropout(self.relu(self.bn1(self.l1(x))), self.drop_rate)
        out2 = f.dropout(self.relu(self.bn2(self.l2(out1))), self.drop_rate)
        out3 = f.dropout(self.relu(self.bn3(self.l3(out2))), self.drop_rate)
        out4 = self.layer_out(out3)

        return out4


def get_model(model_name, num_of_comp_features, comp_hidden_lst, num_classes, dropout):
    model = None

    if model_name == "fc_2_layer":
        model = FC_2_Layer(num_of_comp_features, comp_hidden_lst[0], comp_hidden_lst[1], num_classes, dropout)

    elif model_name == "fc_3_layer":
        model = FC_3_Layer(num_of_comp_features, comp_hidden_lst[0], comp_hidden_lst[1], comp_hidden_lst[2], dropout)

    return model
