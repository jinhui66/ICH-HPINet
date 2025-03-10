# ICH-HPINet: A Hybrid Propagation Interaction Network for Intelligent and Interactive 3D Intracerebral Hemorrhage Segmentation

## Architecture of Our Proposed Model
![image](./figures/main.png)

## File Tree
```
└─data
    -- data_get
└─evaluation
    -- evaluate the metric
└─networks
    -- the model
└─main.py
└─utils.py
└─visual.py
└─eval.py
└─config.py
```

## Dataset
The private dataset contains 286 cases, while the [Physionet](https://physionet.org/content/ct-ich/1.3.1/#files-panel) dataset contains 75 cases. Each case consists of a volume CT data and a volume mask data. During training, the data are splited through five-fold cross-validation.

## Experiment Results
### Comparative Experiment
#### Private dataset
<table align="center">
  <thead>
    <tr>
      <th>Method</th>
      <th>Dice↑</th>
      <th>Jaccard↑</th>
      <th>HD↓</th>
      <th>MAE↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>SAMed</td>
      <td>0.5308</td>
      <td>0.4177</td>
      <td>4.06</td>
      <td>0.0064</td>
    </tr>
    <tr>
      <td>SAM-Med2D</td>
      <td>0.6276</td>
      <td>0.5018</td>
      <td>4.21</td>
      <td>0.0083</td>
    </tr>
    <tr>
      <td>SAM2-UNet</td>
      <td>0.5557</td>
      <td>0.4469</td>
      <td>3.62</td>
      <td>0.0063</td>
    </tr>
    <tr>
      <td>SAMIHS</td>
      <td>0.6681</td>
      <td>0.5504</td>
      <td>3.63</td>
      <td>0.0042</td>
    </tr>
    <tr>
      <td>MedSAM</td>
      <td>0.5932</td>
      <td>0.4650</td>
      <td>4.26</td>
      <td>0.0065</td>
    </tr>
    <tr>
      <td>MedSAM2</td>
      <td>0.5044</td>
      <td>0.3953</td>
      <td>4.07</td>
      <td>0.0069</td>
    </tr>
    <tr>
      <td><strong>ICH-HPINet (Ours)</strong></td>
      <td><strong>0.7797</strong></td>
      <td><strong>0.6532</strong></td>
      <td><strong>3.59</strong></td>
      <td><strong>0.0035</strong></td>
    </tr>
  </tbody>
</table>

#### Physionet dataset
<table align="center">
  <thead>
    <tr>
      <th>Method</th>
      <th>Dice↑</th>
      <th>Jaccard↑</th>
      <th>HD↓</th>
      <th>MAE↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>SAMed</td>
      <td>0.4146</td>
      <td>0.3406</td>
      <td>5.17</td>
      <td>0.0145</td>
    </tr>
    <tr>
      <td>SAM-Med2D</td>
      <td>0.6495</td>
      <td>0.5280</td>
      <td>4.04</td>
      <td>0.0059</td>
    </tr>
    <tr>
      <td>SAM2-UNet</td>
      <td>0.4363</td>
      <td>0.3439</td>
      <td>3.95</td>
      <td>0.0036</td>
    </tr>
    <tr>
      <td>SAMIHS</td>
      <td>0.5819</td>
      <td>0.4546</td>
      <td>3.62</td>
      <td>0.0035</td>
    </tr>
    <tr>
      <td>MedSAM</td>
      <td>0.5292</td>
      <td>0.3978</td>
      <td>5.06</td>
      <td>0.0185</td>
    </tr>
    <tr>
      <td>MedSAM2</td>
      <td>0.6978</td>
      <td>0.5987</td>
      <td>4.00</td>
      <td>0.0055</td>
    </tr>
    <tr>
      <td><strong>ICH-HPINet (Ours)</strong></td>
      <td><strong>0.7467</strong></td>
      <td><strong>0.6094</strong></td>
      <td><strong>2.96</strong></td>
      <td><strong>0.0022</strong></td>
    </tr>
  </tbody>
</table>

### Ablation Experiment
#### Private dataset
<table align="center">
  <thead>
    <tr>
      <th>Method</th>
      <th>Dice↑</th>
      <th>Jaccard↑</th>
      <th>HD↓</th>
      <th>MAE↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Volume-Pro</td>
      <td>0.6554</td>
      <td>0.5255</td>
      <td>5.96</td>
      <td>0.0050</td>
    </tr>
    <tr>
      <td>Slice-Pro</td>
      <td>0.6468</td>
      <td>0.5202</td>
      <td>4.26</td>
      <td>0.0054</td>
    </tr>
    <tr>
      <td><strong>ICH-HPINet (Ours)</strong></td>
      <td><strong>0.7797</strong></td>
      <td><strong>0.6532</strong></td>
      <td><strong>3.59</strong></td>
      <td><strong>0.0035</strong></td>
    </tr>
  </tbody>
</table>

#### Physionet dataset
<table align="center">
  <thead>
    <tr>
      <th>Method</th>
      <th>Dice↑</th>
      <th>Jaccard↑</th>
      <th>HD↓</th>
      <th>MAE↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Volume-Pro</td>
      <td>0.6351</td>
      <td>0.5152</td>
      <td>4.29</td>
      <td>0.0180</td>
    </tr>
    <tr>
      <td>Slice-Pro</td>
      <td>0.4587</td>
      <td>0.3728</td>
      <td>5.45</td>
      <td>0.0148</td>
    </tr>
    <tr>
      <td><strong>ICH-HPINet (Ours)</strong></td>
      <td><strong>0.7467</strong></td>
      <td><strong>0.6094</strong></td>
      <td><strong>2.96</strong></td>
      <td><strong>0.0022</strong></td>
    </tr>
  </tbody>
</table>

## Visual Comparison
![image](./figures/visual.png)