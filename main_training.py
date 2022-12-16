import argparse
from train_FNN_DTI import five_fold_training, training_test, training, test
from extract_features import extract_features, extract_features_train_test

parser = argparse.ArgumentParser(description='FNN arguments')

parser.add_argument(
    '--train',
    type=str,
    default=1,
    metavar='TRAIN',
    help='train(1) or(0) extract(default: 1)')

parser.add_argument(
    '--chln',
    type=str,
    default="1200_300",
    metavar='CHLN',
    help='number of neurons in hidden layers of compound(default: 1200_300)')

parser.add_argument(
    '--lr',
    type=float,
    default=0.0001,
    metavar='LR',
    help='learning rate (default: 0.0001)')

parser.add_argument(
    '--bs',
    type=int,
    default=256,
    metavar='BS',
    help='batch size (default: 256)')

parser.add_argument(
    '--td',
    type=str,
    default="transporter",
    metavar='TD',
    help='the name of the target dataset (default: transporter)')

parser.add_argument(
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
    help='Number of epochs (default: 100)')

parser.add_argument(
    '--sf',
    type=int,
    default=0,
    metavar='SF',
    help='subset flag (default: 0)')

parser.add_argument(
    '--tlf',
    type=int,
    default=0,
    metavar='TLF',
    help='transfer learning flag (default: 0)')

parser.add_argument(
    '--ff',
    type=int,
    default=0,
    metavar='FF',
    help='freeze flag (default: 0)')

parser.add_argument(
    '--fl',
    type=str,
    default="1",
    metavar='FL',
    help='hidden layers to be frozen (default: 1)')

parser.add_argument(
    '--el',
    type=str,
    default="1",
    metavar='EL',
    help='layer to be extracted (default: 0)')

parser.add_argument(
    '--ss',
    type=int,
    default=10,
    metavar='SS',
    help='subset size (default: 10)')

parser.add_argument(
    '--cf',
    type=str,
    default="chemprop",
    metavar='CF',
    help='compound features separated by underscore character (default: chemprop)')

parser.add_argument(
    '--setting',
    type=int,
    default=1,
    metavar='SETTING',
    help='Determines the setting (1: train_val_test, 2:extract layer train_val_test, 3:training_test, 4:only training, '
         '5:extract layer train and test, 6:only test) (default: 1)')

parser.add_argument(
    '--et',
    type=str,
    default="-",
    metavar='ET',
    help='external test dataset (default: -)')

parser.add_argument(
    '--nc',
    type=int,
    default=2,
    metavar='NC',
    help='number of result classes (default: 2)')

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    comp_hidden_layer_neurons = [int(num) for num in args.chln.split("_")]

    if args.setting == 3:
        training_test(args.td, args.sd, args.cf.split("-"), comp_hidden_layer_neurons, args.lr, args.bs,
                      args.model, args.do, args.en, args.epoch, args.sf, args.tlf, args.ff, args.fl, args.ss, args.setting, args.nc)
    elif args.setting == 4:
        training(args.td, args.sd, args.cf.split("-"), comp_hidden_layer_neurons, args.lr, args.bs,
                 args.model, args.do, args.en, args.epoch, args.sf, args.tlf, args.ff, args.fl, args.ss, args.et, args.nc)
    elif args.setting == 5:
        extract_features_train_test(args.td, args.sd, args.cf.split("-"), comp_hidden_layer_neurons, args.lr, args.bs, args.model, args.do, args.en, args.epoch,
                                    args.sf, args.tlf, args.ff, args.fl, args.el, args.ss, args.setting, args.nc)
    elif args.setting == 6:
        test(args.td, args.sd, args.cf.split("-"), comp_hidden_layer_neurons, args.lr, args.bs,
                 args.model, args.do, args.en, args.epoch, args.sf, args.tlf, args.ff, args.fl, args.ss, args.et, args.nc)
    else:
        if args.train == 1:
            five_fold_training(args.td, args.sd, args.cf.split("-"), comp_hidden_layer_neurons, args.lr, args.bs,
                               args.model, args.do, args.en, args.epoch, args.sf, args.tlf, args.ff, args.fl, args.ss, args.setting, args.nc)
        else:
            extract_features(args.td, args.sd, args.cf.split("-"), comp_hidden_layer_neurons, args.lr, args.bs,
                             args.model, args.do, args.en, args.epoch, args.sf, args.tlf, args.ff, args.fl, args.el, args.ss, args.setting)
