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

| Source family |  |  Transporter family (target) training dataset sizes: # of data points (% of the total dataset)  |  |  |  |  |  |  |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|  |  | 5 (0.1%) | 35 (0.5%) | 75 (1%) | 395 (5%) | 795 (10%) | 1985 (25%) | 3980 (50%) |
| GPCR | baseline | 0.160 | 0.312 | 0.329 | 0.446 | 0.510 | 0.525 | 0.531 |
|  | Shallow | 0.178 | 0.321 | 0.338 | 0.459 | 0.501 | 0.520 | 0.526 |
|  | Mode1 | 0.298 | 0.413 | 0.420 | 0.448 | 0.467 | 0.480 | 0.516 |
|  | Mode2 | 0.262 | 0.405 | 0.414 | 0.437 | 0.460 | 0.477 | 0.507 |
|  | Mode3 | 0.118 | 0.323 | 0.376 | 0.467 | 0.505 | 0.522 | 0.527 |
| Ion Channel | baseline | 0.208 | 0.338 | 0.316 | 0.449 | 0.511 | 0.527 | 0.528 |
|  | Shallow | 0.139 | 0.364 | 0.348 | 0.465 | 0.499 | 0.521 | 0.530 |
|  | Mode1 | 0.325 | 0.373 | 0.405 | 0.471 | 0.491 | 0.513 | 0.528 |
|  | Mode2 | 0.307 | 0.365 | 0.391 | 0.461 | 0.492 | 0.511 | 0.523 |
|  | Mode3 | 0.258 | 0.382 | 0.376 | 0.465 | 0.497 | 0.521 | 0.530 |
| Kinase | baseline | 0.129 | 0.326 | 0.315 | 0.449 | 0.511 | 0.525 | 0.532 |
|  | Shallow | 0.224 | 0.315 | 0.357 | 0.462 | 0.500 | 0.515 | 0.531 |
|  | Mode1 | 0.358 | 0.407 | 0.415 | 0.471 | 0.487 | 0.494 | 0.516 |
|  | Mode2 | 0.369 | 0.407 | 0.413 | 0.466 | 0.487 | 0.493 | 0.507 |
|  | Mode3 | 0.233 | 0.355 | 0.397 | 0.479 | 0.507 | 0.525 | 0.528 |
| Nuclear Receptor |  baseline | 0.202 | 0.343 | 0.325 | 0.451 | 0.512 | 0.526 | 0.531 |
|  | Shallow | 0.242 | 0.336 | 0.346 | 0.466 | 0.498 | 0.520 | 0.523 |
|  | Mode1 | 0.250 | 0.346 | 0.381 | 0.455 | 0.485 | 0.496 | 0.522 |
|  | Mode2 | 0.254 | 0.308 | 0.359 | 0.440 | 0.472 | 0.495 | 0.518 |
|  | Mode3 | 0.205 | 0.342 | 0.353 | 0.459 | 0.492 | 0.517 | 0.525 |
| Protease |  baseline | 0.183 | 0.316 | 0.325 | 0.447 | 0.510 | 0.525 | 0.531 |
|  | Shallow | 0.118 | 0.328 | 0.357 | 0.464 | 0.498 | 0.520 | 0.522 |
|  | Mode1 | 0.291 | 0.361 | 0.404 | 0.454 | 0.478 | 0.497 | 0.515 |
|  | Mode2 | 0.310 | 0.340 | 0.400 | 0.449 | 0.468 | 0.491 | 0.516 |
|  | Mode3 | 0.170 | 0.318 | 0.379 | 0.463 | 0.497 | 0.520 | 0.523 |
