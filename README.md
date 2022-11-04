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

## Development and Dependencies

#### [PyTorch 1.12.1](https://pytorch.org/get-started/previous-versions/)
#### [Pandas 1.3.5](https://pandas.pydata.org/pandas-docs/version/1.3.5/getting_started/install.html)
#### [Sklearn 1.1.2](https://scikit-learn.org/1.1/install.html)
#### [Numpy 1.22.4](https://pypi.python.org/pypi/numpy/1.22.4)


## How to re-produce performance comparison results for MDeePred and other methods 
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


#### To create transporter small dataset with size 6
```
python createTrainingandTest.py --d transporter --ss 6
```
#### To obtain baseline peformance results for the same dataset
```
python baseline_training.py --setting 2 --tlf 0 --td transporter --ss 6 --en 0 --sf 1
```
#### To obtain scratch performance result for the same dataset
```
python main_training.py --setting 3 --epoch 50 --ss 6 --en 0 --tlf 0 --sf 1 --td transporter
```
#### To extract the output of the first hidden layer (--el 2 for the output of the second hidden layer )
```
python main_training.py --setting 5 --train 0 --epoch 50 --ss 6 --en 0 --el 1 --tlf 1 --sf 1 --td transporter --sd kinase --nc 2
```
#### To obtain shallow classifier performance result using the output of the first hidden layer (--el 2 for the output of the second hidden layer )
```
python baseline_training.py --setting 2 --tlf 1 --el 1 --td transporter --sd kinase --ss 6 --en 0 --sf 1
```
#### To obtain full fine-tuning performance result for the same dataset
```
python main_training.py --setting 3 --epoch 50 --ss 6 --en 0 --tlf 1 --sf 1 --td transporter --sd kinase --nc 2
```
#### To obtain fine-tuning with freezing layer 1 performance result for the same dataset (--fl 2 with freezing layer 2 )
```
python main_training.py --setting 3 --epoch 50 --ss 6 --en 0 --ff 1 --fl 1 --tlf 1 --ff 1 --sf 1 --td transporter --sd kinase --nc 2
```
#### Output of the scripts
**main_training.py** creates a folder under named **experiment_name** (given as argument **--en**) under **result_files** folder. Two files are created under **results_files/<experiment_name>**: **predictions.txt** contains predictions for independent test dataset. The other one is named as **performance_results.txt** which contains the best performance results for each fold (if setting-1 is chosen) or for the test dataset (if setting-2 is chosen). Sample output files for Davis dataset is given under **results_files/davis_dataset_my_experiment**.

