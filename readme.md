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
└─interact.py
└─config.py
```

## Dataset
The private dataset contains 286 cases, while the [Physionet](https://physionet.org/content/ct-ich/1.3.1/#files-panel) dataset contains 75 cases. Each case consists of a volume CT data and a volume mask data. During training, the data are splited through five-fold cross-validation.

## Train
You can run it through the following command: 
```
python main.py
```

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

Our approach demonstrated significant improvements on both our private and public datasets, enhancing Dice, Jaccard, HD, and MAE scores by at least 0.1116, 0.1028, 0.03, and 0.0007, respectively, on the private dataset, and 0.0489, 0.0107, 0.66 and 0.0013 on the Physionet dataset. Even with fewer annotations, our method consistently achieves optimal results, demonstrating its superiority.

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

The hybrid strategy improved Dice, Jaccard, HD, and MAE scores by at least 0.1243, 0.1277, 0.73, and 0.00143 on the private dataset, and 0.1116, 0.0942, 1.33, and 0.01255 on the Physionet dataset, respectively.

## Visual Comparison
![image](./figures/visual.png)

Most methods perform well in simpler cases, like those in (b), with high segmentation accuracy. However, in more complex scenarios, ICH-HPINet outperforms other methods in terms of precision and competitiveness. For instance, in challenging cases with multiple bleeding points (a and c), our approach shows superior results. In samples with very small outliers such as (d), our method still achieves significant results.

## Interaction Experiment
<table align="center">
  <thead>
    <tr>
      <th>Number</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Dice↑</td>
      <td>0.6133</td>
      <td>0.7763</td>
      <td>0.7770</td>
      <td>0.7831</td>
      <td>0.7820</td>
      <td>0.7856</td>
      <td>0.7868</td>
      <td>0.7865</td>
    </tr>
    <tr>
      <td>Jaccard↑</td>
      <td>0.4499</td>
      <td>0.6471</td>
      <td>0.6469</td>
      <td>0.6537</td>
      <td>0.6527</td>
      <td>0.6567</td>
      <td>0.6583</td>
      <td>0.6580</td>
    </tr>
    <tr>
      <td>HD↓</td>
      <td>4.50</td>
      <td>3.61</td>
      <td>3.61</td>
      <td>3.66</td>
      <td>3.64</td>
      <td>3.60</td>
      <td>3.56</td>
      <td>3.56</td>
    </tr>
    <tr>
      <td>MAE↓</td>
      <td>0.0065</td>
      <td>0.0043</td>
      <td>0.0043</td>
      <td>0.0043</td>
      <td>0.0042</td>
      <td>0.0042</td>
      <td>0.0042</td>
      <td>0.0042</td>
    </tr>
  </tbody>
</table>

You can run it through the following command:
```
python interact.py
```

Tht segmentation results are updated with every interaction, showing the most significant improvement during the initial stages. Following the second interaction, the results begin to stabilize, yet they continue to exhibit overall improvement. These results highlight the effectiveness of our interactive approach and its ability to progressively refine segmentation with additional user input.
