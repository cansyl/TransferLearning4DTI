# Transfer-Learning


## Descriptions of folders and files in the TransferLearning4DTI repository

* **bin** folder includes the source code of TransferLearning4DTI.

* **training_files** folder contains various traininig/test datasets mostly formatted for observational purposes and for employment in future studies
    * **gpcr** contains training data points, features, data splits for GPCR dataset. **compound_feature_vectors/ecfp4.tsv** includes ecfp4 features of ligands. **compound_feature_vectors/chemprop.tsv** includes chemprop features of ligands. **comp_targ_binary.tsv** is a csv file where each line is formatted as <compound_id>,<target_id>,<active/inactive>.  
    * **ionchannel** contains training data points, features, data splits for Ion Channel dataset. **compound_feature_vectors/ecfp4.tsv** includes ecfp4 features of ligands. **compound_feature_vectors/chemprop.tsv** includes chemprop features of ligands. **comp_targ_binary.tsv** is a csv file where each line is formatted as <compound_id>,<target_id>,<active/inactive>.
    * **kinase** contains training data points, features, data splits for Kinase dataset. **compound_feature_vectors/ecfp4.tsv** includes ecfp4 features of ligands. **compound_feature_vectors/chemprop.tsv** includes chemprop features of ligands. **comp_targ_binary.tsv** is a csv file where each line is formatted as <compound_id>,<target_id>,<active/inactive>. 
    * **nuclearreceptor** contains training data points, features, data splits for Nuclear Receptor dataset. **compound_feature_vectors/ecfp4.tsv** includes ecfp4 features of ligands. **compound_feature_vectors/chemprop.tsv** includes chemprop features of ligands. **comp_targ_binary.tsv** is a csv file where each line is formatted as <compound_id>,<target_id>,<active/inactive>. 
    * **protease** contains training data points, features, data splits for Protease dataset. **compound_feature_vectors/ecfp4.tsv** includes ecfp4 features of ligands. **compound_feature_vectors/chemprop.tsv** includes chemprop features of ligands. **comp_targ_binary.tsv** is a csv file where each line is formatted as <compound_id>,<target_id>,<active/inactive>. 
    * **transporter** contains training data points, features, data splits for Kinase dataset based on Setting-2. **compound_feature_vectors/ecfp4.tsv** includes ecfp4 features of ligands. **compound_feature_vectors/chemprop.tsv** includes chemprop features of ligands. **comp_targ_binary.tsv** is a csv file where each line is formatted as <compound_id>,<target_id>,<active/inactive>. 

 * Download the remaining training files [here](https://drive.google.com/drive/folders/1HL8xkWhF8qkRaZleZsp1QW2K1dWYxItd?usp=share_link)

## Development and Dependencies

#### [PyTorch 1.12.1](https://pytorch.org/get-started/previous-versions/)
#### [Pandas 1.3.5](https://pandas.pydata.org/pandas-docs/version/1.3.5/getting_started/install.html)
#### [Sklearn 1.1.2](https://scikit-learn.org/1.1/install.html)
#### [Numpy 1.22.4](https://pypi.python.org/pypi/numpy/1.22.4)
#### [RDKit 2022.9.1](https://www.rdkit.org/docs/Install.html)
#### [Chemprop](https://github.com/chemprop/chemprop#option-1-installing-from-pypi)

## How to re-produce performance comparison results for TransferLearning4DTI 
* Clone the Git Repository

* Run the below commands for each dataset

#### Explanation of Parameters
**--chln**: number of neurons in compound hidden layers (default: 1200_300)

**--lr**:learning rate (default: 0.0001)

**--bs**: batch size (default: 256)

**--td**: the name of the target dataset (default: transporter)

**--sd**: the name of the source dataset (default: kinase)

**--do**: dropout rate (default: 0.1)

**--en**: the name of the experiment (default: my_experiment)

**--model**: model name (default: fc_2_layer)

**--epoch**: number of epochs (default: 100)

**--sf**: subset flag (default: 0)

**--tlf**: transfer learning flag (default: 0)

**--ff**: freeze flag (default: 0)

**--fl**: hidden layer to be frozen (default: 1)

**--el**: layer to be extracted (default: 0)

**--ss**: subset size (default: 10)

**--cf**: compound features separated by underscore character (default: chemprop)

**--setting**: Determines the setting (1: train_val_test, 2:extract layer train_val_test, 3:training_test, 4:only training, 5:extract layer train and test) (default: 1)

**--et**: external test dataset (default: -)

**--nc**: number of result classes (default: 2)

**--train**: (1) train or (0) extract features(default: 1)


#### Option 1 
#### to reproduce performance results 
#### create a Transporter small dataset with a size 6
```
python create_small_dataset.py --d transporter --ss 6
```
#### obtain baseline performance results for the created dataset
```
python baseline_training.py --setting 2 --tlf 0 --td transporter --ss 6 --en 0 --sf 1
``` 
#### obtain scratch performance results for the same dataset
```
python main_training.py --setting 3 --epoch 50 --ss 6 --en 0 --tlf 0 --sf 1 --td transporter
``` 
#### extract the output of the first hidden layer (--el 2 for the output of the second hidden layer )
```
python main_training.py --setting 5 --train 0 --epoch 50 --ss 6 --en 0 --el 1 --tlf 1 --sf 1 --td transporter --sd kinase
``` 
#### obtain shallow classifier performance results using the output of the first hidden layer (--el 2 for the output of the second hidden layer )
```
python baseline_training.py --setting 2 --tlf 1 --el 1 --td transporter --sd kinase --ss 6 --en 0 --sf 1
``` 
#### obtain full fine-tuning performance results for the same dataset
```
python main_training.py --setting 3 --epoch 50 --ss 6 --en 0 --tlf 1 --sf 1 --td transporter --sd kinase
``` 
#### obtain fine-tuning with freezing layer 1 performance result for the same dataset (--fl 2 with freezing layer 2 )
```
python main_training.py --setting 3 --epoch 50 --ss 6 --en 0 --ff 1 --fl 1 --tlf 1 --ff 1 --sf 1 --td transporter --sd kinase
```
#### Option 2 - Fine-tune your training dataset
#### extract features using the chemprop tool for your training dataset by running the following command
**--trainisc**: the name of the training file. it should contains id, smiles and compound columns (default: input/train.csv)

**--name**: the name of the your protein or protein family (default: new_family)

**--sc**: the name of the source checkpoint (default: kinase)
```
python convert_chemprop_ftune.py --trainisc input/train.csv --name your_family_name --sc kinase
```
#### train your dataset by running the following command without transfer learning
```
python main_training.py --setting 4 --td your_family_name 
```
#### train your dataset by running the following command with transfer learning
```
python main_training.py --setting 4 --td your_family_name --sd kinase --tlf 1
```
#### Option 3 Predict your test dataset
#### extract features using the chemprop tool for your test dataset by running the following command (it will create test_chemprop file under output directory)
**--testisf**: the name of the test file. it should contains id and smiles columns (default: input/test.csv)

**--sc**: the name of the source checkpoint (default: kinase)
```
python convert_chemprop_predict.py --testisf input/test.csv --sc kinase
```
#### get predictions for your test dataset without training you can run the following command
```
python main_training.py --setting 6 --sd kinase --et output/test_chemprop.tsv --tlf 1
```
#### get predictions for your test dataset by training you can run the following command
```
python main_training.py --setting 4 --td transporter --sd kinase --et output/test_chemprop.tsv --tlf 1
```
#### Option 4
#### Fine-tune your training dataset and get predictions for your test dataset using the fine-tuned model (training and testing)
#### extract features using the chemprop tool for your training and test dataset by running the following command (it will create test_chemprop file under output directory)
**--trainisc**: the name of the training file. it should contains id, smiles and compound columns (default: input/train.csv)

**--name**: the name of the your protein or protein family (default: new_family)

**--testisf**: the name of the test file. it should contains id and smiles columns (default: input/test.csv)

**--sc**: the name of the source checkpoint (default: kinase)

```
python convert_chemprop_ftune_predict.py --trainisc input/train.csv --name your_family_name --sc kinase --testisf input/test.csv
```
#### To get predictions for your test dataset you can run the following command
#### full fine-tune predictions
```
python main_training.py --setting 4 --td your_family_name --sd kinase --et output/test_chemprop.tsv --tlf 1
```
#### fine-tune with freeze predictions
```
python main_training.py --setting 4 --td your_family_name --sd kinase --et output/test_chemprop.tsv --tlf 1 --ff 1 --fl 1
```
#### Output of the scripts
**main_training.py** creates a folder under named **experiment_name** (given as argument **--en**) under **result_files** folder. One file is created under **results_files/<experiment_name>**: **performance_results.txt** which contains the best performance results for test dataset. Sample output files for Transporter dataset is given under **results_files/transporter**.

#### Results
| 4,000     | scratch       | shallow       | Mode 1        | Mode 2        | Mode 3        |
|-----------|---------------|---------------|---------------|---------------|---------------|
| MCC       | 0.531 + 0.005 | 0.533 + 0.008 | 0.522 + 0.010 | 0.518 + 0.008 | 0.534 + 0.011 |
| AUROC     | 0.770 + 0.003 | 0.771 + 0.004 | 0.764 + 0.005 | 0.763 + 0.005 | 0.771 + 0.005 |
| Precision | 0.684 + 0.013 | 0.680 + 0.006 | 0.682 + 0.017 | 0.681 + 0.018 | 0.690 + 0.015 |
| Recall    | 0.763 + 0.023 | 0.773 + 0.009 | 0.752 + 0.028 | 0.751 + 0.018 | 0.768 + 0.010 |
| F1-Score  | 0.721 + 0.005 | 0.724 + 0.005 | 0.715 + 0.007 | 0.712 + 0.007 | 0.723 + 0.006 |
| Accuracy  | 0.771 + 0.004 | 0.771 + 0.004 | 0.767 + 0.007 | 0.766 + 0.006 | 0.772 + 0.005 |
|           |               |               |               |               |               |
| 1,000     | scratch       | shallow       | Mode 1        | Mode 2        | Mode 3        |
| MCC       | 0.481 + 0.014 | 0.517 + 0.014 | 0.494 + 0.011 | 0.489 + 0.011 | 0.523 + 0.013 |
| AUROC     | 0.745 + 0.007 | 0.765 + 0.007 | 0.751 + 0.005 | 0.749 + 0.006 | 0.768 + 0.007 |
| Precision | 0.644 + 0.014 | 0.661 + 0.013 | 0.661 + 0.012 | 0.659 + 0.009 | 0.667 + 0.014 |
| Recall    | 0.756 + 0.025 | 0.786 + 0.020 | 0.744 + 0.018 | 0.740 + 0.019 | 0.788 + 0.017 |
| F1-Score  | 0.695 + 0.009 | 0.717 + 0.008 | 0.700 + 0.006 | 0.697 + 0.008 | 0.720 + 0.007 |
| Accuracy  | 0.743 + 0.008 | 0.760 + 0.008 | 0.753 + 0.007 | 0.751 + 0.005 | 0.763 + 0.007 |
|           |               |               |               |               |               |
| 400       | scratch       | shallow       | Mode 1        | Mode 2        | Mode 3        |
| MCC       | 0.400 + 0.008 | 0.488 + 0.019 | 0.469 + 0.015 | 0.464 + 0.015 | 0.495 + 0.016 |
| AUROC     | 0.705 + 0.004 | 0.750 + 0.010 | 0.739 + 0.007 | 0.736 + 0.008 | 0.753 + 0.008 |
| Precision | 0.584 + 0.011 | 0.647 + 0.014 | 0.645 + 0.017 | 0.639 + 0.016 | 0.649 + 0.014 |
| Recall    | 0.749 + 0.029 | 0.778 + 0.026 | 0.734 + 0.027 | 0.738 + 0.029 | 0.788 + 0.025 |
| F1-Score  | 0.656 + 0.006 | 0.702 + 0.011 | 0.686 + 0.009 | 0.684 + 0.010 | 0.706 + 0.009 |
| Accuracy  | 0.695 + 0.007 | 0.744 + 0.010 | 0.740 + 0.009 | 0.736 + 0.009 | 0.748 + 0.010 |
|           |               |               |               |               |               |
| 96        | scratch       | shallow       | Mode 1        | Mode 2        | Mode 3        |
| MCC       | 0.383 + 0.017 | 0.410 + 0.031 | 0.427 + 0.030 | 0.423 + 0.028 | 0.419 + 0.027 |
| AUROC     | 0.694 + 0.009 | 0.709 + 0.017 | 0.718 + 0.016 | 0.716 + 0.014 | 0.714 + 0.014 |
| Precision | 0.551 + 0.013 | 0.609 + 0.020 | 0.614 + 0.023 | 0.612 + 0.022 | 0.610 + 0.021 |
| Recall    | 0.807 + 0.031 | 0.755 + 0.062 | 0.726 + 0.051 | 0.723 + 0.044 | 0.770 + 0.064 |
| F1-Score  | 0.654 + 0.009 | 0.659 + 0.021 | 0.664 + 0.020 | 0.662 + 0.018 | 0.663 + 0.019 |
| Accuracy  | 0.669 + 0.013 | 0.709 + 0.013 | 0.716 + 0.016 | 0.714 + 0.015 | 0.710 + 0.015 |
|           |               |               |               |               |               |
| 48        | scratch       | shallow       | Mode 1        | Mode 2        | Mode 3        |
| MCC       | 0.368 + 0.025 | 0.373 + 0.041 | 0.410 + 0.033 | 0.405 + 0.031 | 0.385 + 0.035 |
| AUROC     | 0.685 + 0.014 | 0.689 + 0.022 | 0.709 + 0.017 | 0.706 + 0.016 | 0.696 + 0.018 |
| Precision | 0.541 + 0.018 | 0.590 + 0.032 | 0.597 + 0.031 | 0.594 + 0.031 | 0.586 + 0.029 |
| Recall    | 0.807 + 0.050 | 0.725 + 0.096 | 0.738 + 0.060 | 0.735 + 0.056 | 0.740 + 0.105 |
| F1-Score  | 0.647 + 0.015 | 0.630 + 0.028 | 0.657 + 0.020 | 0.655 + 0.018 | 0.643 + 0.023 |
| Accuracy  | 0.658 + 0.019 | 0.690 + 0.022 | 0.702 + 0.022 | 0.700 + 0.021 | 0.690 + 0.022 |
|           |               |               |               |               |               |
| 12        | scratch       | shallow       | Mode 1        | Mode 2        | Mode 3        |
| MCC       | 0.293 + 0.082 | 0.280 + 0.080 | 0.367 + 0.032 | 0.370 + 0.029 | 0.307 + 0.062 |
| AUROC     | 0.638 + 0.048 | 0.640 + 0.041 | 0.686 + 0.017 | 0.687 + 0.015 | 0.652 + 0.034 |
| Precision | 0.490 + 0.044 | 0.536 + 0.055 | 0.563 + 0.034 | 0.575 + 0.035 | 0.543 + 0.050 |
| Recall    | 0.856 + 0.073 | 0.675 + 0.167 | 0.745 + 0.079 | 0.725 + 0.082 | 0.687 + 0.135 |
| F1-Score  | 0.619 + 0.028 | 0.577 + 0.053 | 0.637 + 0.022 | 0.634 + 0.023 | 0.596 + 0.046 |
| Accuracy  | 0.589 + 0.069 | 0.638 + 0.047 | 0.673 + 0.025 | 0.680 + 0.021 | 0.645 + 0.045 |
|           |               |               |               |               |               |
| 6         | scratch       | shallow       | Mode 1        | Mode 2        | Mode 3        |
| MCC       | 0.227 + 0.109 | 0.214 + 0.130 | 0.367 + 0.029 | 0.372 + 0.027 | 0.269 + 0.098 |
| AUROC     | 0.600 + 0.061 | 0.605 + 0.065 | 0.686 + 0.015 | 0.688 + 0.014 | 0.632 + 0.050 |
| Precision | 0.460 + 0.051 | 0.494 + 0.071 | 0.556 + 0.031 | 0.559 + 0.029 | 0.515 + 0.060 |
| Recall    | 0.889 + 0.088 | 0.694 + 0.209 | 0.764 + 0.076 | 0.768 + 0.075 | 0.734 + 0.163 |
| F1-Score  | 0.600 + 0.029 | 0.551 + 0.074 | 0.640 + 0.018 | 0.643 + 0.018 | 0.585 + 0.061 |
| Accuracy  | 0.536 + 0.091 | 0.594 + 0.076 | 0.668 + 0.025 | 0.671 + 0.022 | 0.615 + 0.065 |
|           |               |               |               |               |               |
| 2         | scratch       | shallow       | Mode 1        | Mode 2        | Mode 3        |
| MCC       | 0.200 + 0.122 | 0.126 + 0.168 | 0.365 + 0.023 | 0.369 + 0.023 | 0.227 + 0.124 |
| AUROC     | 0.589 + 0.065 | 0.558 + 0.081 | 0.683 + 0.013 | 0.685 + 0.013 | 0.603 + 0.065 |
| Precision | 0.454 + 0.053 | 0.453 + 0.107 | 0.554 + 0.033 | 0.558 + 0.035 | 0.500 + 0.087 |
| Recall    | 0.883 + 0.111 | 0.622 + 0.275 | 0.767 + 0.088 | 0.773 + 0.087 | 0.692 + 0.254 |
| F1-Score  | 0.592 + 0.033 | 0.490 + 0.127 | 0.638 + 0.018 | 0.640 + 0.017 | 0.543 + 0.111 |
| Accuracy  | 0.523 + 0.098 | 0.548 + 0.088 | 0.665 + 0.026 | 0.668 + 0.027 | 0.583 + 0.086 |
