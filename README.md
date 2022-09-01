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




| Family | # of Targets | # of active compounds | # of inactive compounds | # of DTI | training dataset size | test dataset size |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| GPCR | 230 | 36,924 | 31,085 | 68,009 | 56,675 | 11,334 |
| Ion Channel | 118 | 5,996 | 14,167 | 20,163 | 16,803 | 3,360 |
| Kinase | 318 | 35,531 | 30,778 | 66,309 | 55,259 | 11,050 |
| Nuclear Receptor | 37 | 5,099 | 6,668 | 11,767 | 9,807 | 1,960 |
| Protease | 177 | 15,718 | 19,518 | 35,236 | 29,364 | 5,872 |
| Transporter | 92 | 3,666 | 5,898 | 9,564 | 7,970 | 1,594 |


<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" rowspan="2">Source family</th>
    <th class="tg-c3ow" rowspan="2"><br><br></th>
    <th class="tg-c3ow" colspan="7"><br>Transporter family (target) training dataset sizes: # of data points (% of the total dataset) </th>
  </tr>
  <tr>
    <th class="tg-c3ow">5 (0.1%)</th>
    <th class="tg-c3ow">35 (0.5%)</th>
    <th class="tg-c3ow">75 (1%)</th>
    <th class="tg-c3ow">395 (5%)</th>
    <th class="tg-c3ow">795 (10%)</th>
    <th class="tg-c3ow">1985 (25%)</th>
    <th class="tg-c3ow">3980 (50%)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow" rowspan="5">GPCR</td>
    <td class="tg-c3ow">baseline</td>
    <td class="tg-dvpl">0.160</td>
    <td class="tg-dvpl">0.312</td>
    <td class="tg-dvpl">0.329</td>
    <td class="tg-dvpl">0.446</td>
    <td class="tg-dvpl">0.510</td>
    <td class="tg-dvpl">0.525</td>
    <td class="tg-dvpl">0.531</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Shallow</td>
    <td class="tg-dvpl">0.178</td>
    <td class="tg-dvpl">0.321</td>
    <td class="tg-dvpl">0.338</td>
    <td class="tg-dvpl">0.459</td>
    <td class="tg-dvpl">0.501</td>
    <td class="tg-dvpl">0.520</td>
    <td class="tg-dvpl">0.526</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Mode1</td>
    <td class="tg-dvpl">0.298</td>
    <td class="tg-dvpl">0.413</td>
    <td class="tg-dvpl">0.420</td>
    <td class="tg-dvpl">0.448</td>
    <td class="tg-dvpl">0.467</td>
    <td class="tg-dvpl">0.480</td>
    <td class="tg-dvpl">0.516</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Mode2</td>
    <td class="tg-dvpl">0.262</td>
    <td class="tg-dvpl">0.405</td>
    <td class="tg-dvpl">0.414</td>
    <td class="tg-dvpl">0.437</td>
    <td class="tg-dvpl">0.460</td>
    <td class="tg-dvpl">0.477</td>
    <td class="tg-dvpl">0.507</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Mode3</td>
    <td class="tg-dvpl">0.118</td>
    <td class="tg-dvpl">0.323</td>
    <td class="tg-dvpl">0.376</td>
    <td class="tg-dvpl">0.467</td>
    <td class="tg-dvpl">0.505</td>
    <td class="tg-dvpl">0.522</td>
    <td class="tg-dvpl">0.527</td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="5">Ion Channel</td>
    <td class="tg-c3ow">baseline</td>
    <td class="tg-dvpl">0.208</td>
    <td class="tg-dvpl">0.338</td>
    <td class="tg-dvpl">0.316</td>
    <td class="tg-dvpl">0.449</td>
    <td class="tg-dvpl">0.511</td>
    <td class="tg-dvpl">0.527</td>
    <td class="tg-dvpl">0.528</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Shallow</td>
    <td class="tg-dvpl">0.139</td>
    <td class="tg-dvpl">0.364</td>
    <td class="tg-dvpl">0.348</td>
    <td class="tg-dvpl">0.465</td>
    <td class="tg-dvpl">0.499</td>
    <td class="tg-dvpl">0.521</td>
    <td class="tg-dvpl">0.530</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Mode1</td>
    <td class="tg-dvpl">0.325</td>
    <td class="tg-dvpl">0.373</td>
    <td class="tg-dvpl">0.405</td>
    <td class="tg-dvpl">0.471</td>
    <td class="tg-dvpl">0.491</td>
    <td class="tg-dvpl">0.513</td>
    <td class="tg-dvpl">0.528</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Mode2</td>
    <td class="tg-dvpl">0.307</td>
    <td class="tg-dvpl">0.365</td>
    <td class="tg-dvpl">0.391</td>
    <td class="tg-dvpl">0.461</td>
    <td class="tg-dvpl">0.492</td>
    <td class="tg-dvpl">0.511</td>
    <td class="tg-dvpl">0.523</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Mode3</td>
    <td class="tg-dvpl">0.258</td>
    <td class="tg-dvpl">0.382</td>
    <td class="tg-dvpl">0.376</td>
    <td class="tg-dvpl">0.465</td>
    <td class="tg-dvpl">0.497</td>
    <td class="tg-dvpl">0.521</td>
    <td class="tg-dvpl">0.530</td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="5">Kinase</td>
    <td class="tg-c3ow">baseline</td>
    <td class="tg-dvpl">0.129</td>
    <td class="tg-dvpl">0.326</td>
    <td class="tg-dvpl">0.315</td>
    <td class="tg-dvpl">0.449</td>
    <td class="tg-dvpl">0.511</td>
    <td class="tg-dvpl">0.525</td>
    <td class="tg-dvpl">0.532</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Shallow</td>
    <td class="tg-dvpl">0.224</td>
    <td class="tg-dvpl">0.315</td>
    <td class="tg-dvpl">0.357</td>
    <td class="tg-dvpl">0.462</td>
    <td class="tg-dvpl">0.500</td>
    <td class="tg-dvpl">0.515</td>
    <td class="tg-dvpl">0.531</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Mode1</td>
    <td class="tg-dvpl">0.358</td>
    <td class="tg-dvpl">0.407</td>
    <td class="tg-dvpl">0.415</td>
    <td class="tg-dvpl">0.471</td>
    <td class="tg-dvpl">0.487</td>
    <td class="tg-dvpl">0.494</td>
    <td class="tg-dvpl">0.516</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Mode2</td>
    <td class="tg-dvpl">0.369</td>
    <td class="tg-dvpl">0.407</td>
    <td class="tg-dvpl">0.413</td>
    <td class="tg-dvpl">0.466</td>
    <td class="tg-dvpl">0.487</td>
    <td class="tg-dvpl">0.493</td>
    <td class="tg-dvpl">0.507</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Mode3</td>
    <td class="tg-dvpl">0.233</td>
    <td class="tg-dvpl">0.355</td>
    <td class="tg-dvpl">0.397</td>
    <td class="tg-dvpl">0.479</td>
    <td class="tg-dvpl">0.507</td>
    <td class="tg-dvpl">0.525</td>
    <td class="tg-dvpl">0.528</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="5">Nuclear Receptor</td>
    <td class="tg-0pky"> baseline</td>
    <td class="tg-dvpl">0.202</td>
    <td class="tg-dvpl">0.343</td>
    <td class="tg-dvpl">0.325</td>
    <td class="tg-dvpl">0.451</td>
    <td class="tg-dvpl">0.512</td>
    <td class="tg-dvpl">0.526</td>
    <td class="tg-dvpl">0.531</td>
  </tr>
  <tr>
    <td class="tg-0pky">Shallow</td>
    <td class="tg-dvpl">0.242</td>
    <td class="tg-dvpl">0.336</td>
    <td class="tg-dvpl">0.346</td>
    <td class="tg-dvpl">0.466</td>
    <td class="tg-dvpl">0.498</td>
    <td class="tg-dvpl">0.520</td>
    <td class="tg-dvpl">0.523</td>
  </tr>
  <tr>
    <td class="tg-0pky">Mode1</td>
    <td class="tg-dvpl">0.250</td>
    <td class="tg-dvpl">0.346</td>
    <td class="tg-dvpl">0.381</td>
    <td class="tg-dvpl">0.455</td>
    <td class="tg-dvpl">0.485</td>
    <td class="tg-dvpl">0.496</td>
    <td class="tg-dvpl">0.522</td>
  </tr>
  <tr>
    <td class="tg-0pky">Mode2</td>
    <td class="tg-dvpl">0.254</td>
    <td class="tg-dvpl">0.308</td>
    <td class="tg-dvpl">0.359</td>
    <td class="tg-dvpl">0.440</td>
    <td class="tg-dvpl">0.472</td>
    <td class="tg-dvpl">0.495</td>
    <td class="tg-dvpl">0.518</td>
  </tr>
  <tr>
    <td class="tg-0pky">Mode3</td>
    <td class="tg-dvpl">0.205</td>
    <td class="tg-dvpl">0.342</td>
    <td class="tg-dvpl">0.353</td>
    <td class="tg-dvpl">0.459</td>
    <td class="tg-dvpl">0.492</td>
    <td class="tg-dvpl">0.517</td>
    <td class="tg-dvpl">0.525</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="5">Protease</td>
    <td class="tg-0pky"> baseline</td>
    <td class="tg-dvpl">0.183</td>
    <td class="tg-dvpl">0.316</td>
    <td class="tg-dvpl">0.325</td>
    <td class="tg-dvpl">0.447</td>
    <td class="tg-dvpl">0.510</td>
    <td class="tg-dvpl">0.525</td>
    <td class="tg-dvpl">0.531</td>
  </tr>
  <tr>
    <td class="tg-0pky">Shallow</td>
    <td class="tg-dvpl">0.118</td>
    <td class="tg-dvpl">0.328</td>
    <td class="tg-dvpl">0.357</td>
    <td class="tg-dvpl">0.464</td>
    <td class="tg-dvpl">0.498</td>
    <td class="tg-dvpl">0.520</td>
    <td class="tg-dvpl">0.522</td>
  </tr>
  <tr>
    <td class="tg-0pky">Mode1</td>
    <td class="tg-dvpl">0.291</td>
    <td class="tg-dvpl">0.361</td>
    <td class="tg-dvpl">0.404</td>
    <td class="tg-dvpl">0.454</td>
    <td class="tg-dvpl">0.478</td>
    <td class="tg-dvpl">0.497</td>
    <td class="tg-dvpl">0.515</td>
  </tr>
  <tr>
    <td class="tg-0pky">Mode2</td>
    <td class="tg-dvpl">0.310</td>
    <td class="tg-dvpl">0.340</td>
    <td class="tg-dvpl">0.400</td>
    <td class="tg-dvpl">0.449</td>
    <td class="tg-dvpl">0.468</td>
    <td class="tg-dvpl">0.491</td>
    <td class="tg-dvpl">0.516</td>
  </tr>
  <tr>
    <td class="tg-0pky">Mode3</td>
    <td class="tg-dvpl">0.170</td>
    <td class="tg-dvpl">0.318</td>
    <td class="tg-dvpl">0.379</td>
    <td class="tg-dvpl">0.463</td>
    <td class="tg-dvpl">0.497</td>
    <td class="tg-dvpl">0.520</td>
    <td class="tg-dvpl">0.523</td>
  </tr>
</tbody>
</table>
