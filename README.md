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

#### Option 2
#### Fine-tune your training dataset and get predictions for your test dataset using the fine-tuned model (training and testing)
#### extract features using the chemprop tool for your training dataset please run the following command
```
chemprop_fingerprint --test_path input/train_smiles_class.csv --checkpoint_path chemprop/kinase_checkpoints/model.pt --preds_path output/train_smiles_chemprop.csv
```
#### convert the output of the chemprop to the right format you can run the following command (you should have a file called train_id_smiles.csv and it should contain the id and smiles for each compound)
```
python chemprop_train_convert.py --tis input/train_id_smiles.csv --tsc output/train_smiles_chemprop.csv --scf input/train_smiles_class.csv --name new_family
```
#### extract features using chemprop tool for your test dataset please run the following command (you need to specify one of the six checkpoints, the default option is kinase)
```
chemprop_fingerprint --test_path input/test_smiles_class.csv --checkpoint_path chemprop/kinase_checkpoints/model.pt --preds_path output/test_smiles_chemprop.csv
```
#### convert the output of the chemprop to the right format you can run the following command (you should have a file called test_id_smiles.csv and it should contain the id and smiles for each compound)
```
python chemprop_test_convert.py --tis input/test_id_smiles.csv --tsc output/test_smiles_chemprop.csv
```
#### To get predictions for your test dataset you can run the following command
#### full fine-tune predictions
```
python main_training.py --setting 4 --td new_family --sd kinase --et output/test_smiles_chemprop.tsv --tlf 1
```
#### fine-tune with freeze predictions
```
python main_training.py --setting 4 --td new_family --sd kinase --et output/test_smiles_chemprop.tsv --tlf 1 --ff 1 --fl 1
```

#### Option 3
#### Get predictions for your test dataset using one of the six pre-trained models following the below steps. (no training, only testing)
#### extract features using chemprop tool for your test dataset please run the following command (you need to specify one of the six checkpoints, the default option is kinase)
```
chemprop_fingerprint --test_path input/test_smiles_class.csv --checkpoint_path chemprop/kinase_checkpoints/model.pt --preds_path output/test_smiles_chemprop.csv
```
#### convert the output of the chemprop to the right format you can run the following command (you should have a file called test_id_smiles.csv and it should contain the id and smiles for each compound)
```
python chemprop_test_convert.py --tis input/test_id_smiles.csv --tsc output/test_smiles_chemprop.csv
```
#### get predictions for your test dataset without training you can run the following command
```
python main_training.py --setting 6 --sd kinase --et output/test_smiles_chemprop.tsv --tlf 1
```
#### get predictions for your test dataset without training you can run the following command
```
python main_training.py --setting 4 --td transporter --sd kinase --et output/test_smiles_chemprop.tsv --tlf 1
```
#### Output of the scripts
**main_training.py** creates a folder under named **experiment_name** (given as argument **--en**) under **result_files** folder. Two files are created under **results_files/<experiment_name>**: **predictions.txt** contains predictions for independent test dataset. The other one is named as **performance_results.txt** which contains the best performance results for each fold (if setting-1 is chosen) or for the test dataset (if setting-2 is chosen). Sample output files for Davis dataset is given under **results_files/davis_dataset_my_experiment**.

