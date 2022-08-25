# Transfer-Learning

Grid search for hyper-parameter values is performed over the number of hidden layers {1, 2, 3, 4}, the hidden layer size {64, 128, 256, 512, 1024, 2048, 4096}, the learning rate {0.01, 0,001, 0.0001}, the dropout rate {0.1, 0.2, 0.3, 0.4, 0.5}, the number of training epochs {20, 40, 50, 80, 100}, and the batch sizes {64, 128, 256, 512}.

 
![figure1](https://user-images.githubusercontent.com/1288719/164443353-addc0237-6b48-45b0-9cf1-da02d3896ce1.png)
Figure 1: An FNN is composed of roughly two parts: lower layers where feature extraction is performed and upper layer(s) where classification is performed.

![figure2](https://user-images.githubusercontent.com/1288719/164443975-c32f37a5-d1c4-43f6-ab08-e8221727a434.png)
Figure 2: A source model is obtained by training the model with a sufficient number of source training data.

![figure3](https://user-images.githubusercontent.com/1288719/164443978-791799c1-7e97-4019-954f-355e02c6c247.png)
Figure 3: Mode 1, Full fine-tuning: the values of weights of the trained source model are used as the initial values of the target model and it is trained with the target training data.

![figure4](https://user-images.githubusercontent.com/1288719/164443365-275234c8-5a18-4481-b219-6cfd095d4e37.png)
Figure 4: Mode 2, A few bottom layers (the first part, which is for feature extraction) of the trained source model are frozen; that is, their weights are not updated during training (backpropagation) with the target training data.

![figure5](https://user-images.githubusercontent.com/1288719/164443982-f6a2f7af-cb8b-4f6f-8018-d29310dbed8d.png)
Figure 5: Mode 3, A shallow classifier is used instead of the classification layer of our source model. For this, we only trained the shallow classifier with the target dataset. Mode 3 is similar to Mode 2 except that a shallow classifier is used instead of the second part of the FNN.

![figure6](https://user-images.githubusercontent.com/1288719/164443309-6f50f203-cfe1-40c7-b09d-ebb32982f16e.png)
Figure 6: Training loss comparison between training from scratch vs full fine-tune for the Transporter family (10%).



![dataset_size](https://user-images.githubusercontent.com/1288719/186678429-611820c8-54a5-416f-87ad-e80c9dc92382.png)
![results](https://user-images.githubusercontent.com/1288719/186680015-19905931-3bf2-4cd5-8d08-1fe482b29ea0.png)
