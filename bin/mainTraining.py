import argparse
from feedForwardDNNDTIFiveFold import five_fold_training, training_test, training
from extractFeatureDNNDTIV2 import extract_features


parser = argparse.ArgumentParser(description='feedForwardDNN arguments')

parser.add_argument(
    '--train',
    type=str,
    default=1,
    metavar='TRAIN',
    help='train or extract(default: 1)')

parser.add_argument(
    '--chln',
    type=str,
    default="4096_1024",
    metavar='HLN',
    help='number of neurons in hidden layers of compound(default: 4096_1024)')

parser.add_argument(
    '--thln',
    type=str,
    default="40_10",
    metavar='HLN',
    help='number of neurons in hidden layers of target(default: 40_10)')

parser.add_argument(
    '--lr',
    type=float,
    default=0.0001,
    metavar='LR',
    help='learning rate (default: 0.0001)')

parser.add_argument(
    # '--batch-size',
    '--bs',
    type=int,
    default=256,
    metavar='BS',
    help='batch size (default: 256)')

parser.add_argument(
    # '--target-data',
    '--td',
    type=str,
    default="transporter",
    metavar='TD',
    help='the name of the target dataset (default: transporter)')

parser.add_argument(
    # '--source-data',
    '--sd',
    type=str,
    default="kinase",
    metavar='SD',
    help='the name of the source dataset (default: kinase)')

parser.add_argument(
    '--do',
    type=float,
    default=0.1,
    metavar='DO',
    help='dropout rate (default: 0.1)')

parser.add_argument(
    '--en',
    type=str,
    default="my_experiments",
    metavar='EN',
    help='the name of the experiment (default: my_experiment)')

parser.add_argument(
    '--model',
    type=str,
    default="fc_2_layer",
    metavar='mn',
    help='model name (default: fc_2_layer)')

parser.add_argument(
    '--epoch',
    type=int,
    default=100,
    metavar='EPOCH',
    help='Number of epochs (default: 50)')

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
    # '--freeze-flag',
    '--ff',
    type=int,
    default=0,
    metavar='FF',
    help='freeze flag (default: 0)')

parser.add_argument(
    # '--frozen-layers`',
    '--fl',
    type=str,
    default="1",
    metavar='FL',
    help='hidden layers to be frozen (default: 1)')

parser.add_argument(
    # '--extracted-layer`',
    '--el',
    type=str,
    default="1",
    metavar='EL',
    help='layer to be extracted (default: 0)')

parser.add_argument(
    # '--subset-size`',
    '--ss',
    type=int,
    default=10,
    metavar='SS',
    help='subset size (default: 10)')

parser.add_argument(
    # '--compound-features',
    '--cf',
    type=str,
    default="ecfp4",
    metavar='CF',
    help='compound features separated by underscore character (default: ecfp4)')

parser.add_argument(
    # '--target-features',
    '--tf',
    type=str,
    default="AAC",
    metavar='TF',
    help='target features separated by underscore character (default: AAC)')

parser.add_argument(
    # '--setting',
    '--setting',
    type=int,
    default=1,
    metavar='SETTING',
    help='Determines the setting (1: train_val_test, 2:after removing some targets, 3:training_test, 4:only training) (default: 1)')

parser.add_argument(
    # '--external-test',
    '--et',
    type=str,
    default="-",
    metavar='ET',
    help='external test dataset (default: -)')

parser.add_argument(
    # '--input-path`',
    '--ip',
    type=str,
    default="/home/adalkiran/PycharmProjects/mainProteinFamilyClassification",
    metavar='IP',
    help='input path (default: /home/adalkiran/PycharmProjects/mainProteinFamilyClassification)')

parser.add_argument(
    # '--output-path`',
    '--op',
    type=str,
    default="/home/adalkiran/PycharmProjects/mainProteinFamilyClassification",
    metavar='OP',
    help='output path (default: /home/adalkiran/PycharmProjects/mainProteinFamilyClassification)')


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    comp_hidden_layer_neurons = [int(num) for num in args.chln.split("_")]
    tar_hidden_layer_neurons = [int(num) for num in args.thln.split("_")]

    if args.setting == 3:
        training_test(args.td, args.sd, (args.cf).split("_"), (args.tf).split("_"), comp_hidden_layer_neurons, tar_hidden_layer_neurons, args.lr, args.bs,
                               args.model, args.do, args.en, args.epoch, args.sf, args.tlf, args.ff, args.fl, args.ss,
                               args.ip, args.op, args.setting)
    if args.setting == 4:
        training(args.td, args.sd, (args.cf).split("_"), (args.tf).split("_"), comp_hidden_layer_neurons, tar_hidden_layer_neurons, args.lr, args.bs,
                               args.model, args.do, args.en, args.epoch, args.sf, args.tlf, args.ff, args.fl, args.ss,
                               args.ip, args.op, args.et)
    else:
        if args.train == 1:
            five_fold_training(args.td, args.sd, (args.cf).split("_"), (args.tf).split("_"), comp_hidden_layer_neurons, tar_hidden_layer_neurons, args.lr, args.bs,
                               args.model, args.do, args.en, args.epoch, args.sf, args.tlf, args.ff, args.fl, args.ss,
                               args.ip, args.op, args.setting)
        else:
            extract_features(args.td, args.sd, (args.cf).split("_"), (args.tf).split("_"), comp_hidden_layer_neurons, tar_hidden_layer_neurons, args.lr, args.bs,
                         args.model, args.do, args.en, args.epoch, args.sf, args.tlf, args.ff, args.fl, args.el, args.ss,
                         args.ip, args.setting)
