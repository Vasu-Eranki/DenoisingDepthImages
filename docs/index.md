# Abstract


<ins>Objective</ins>  
<p align ="justify">Depth maps are a critical part of many computer vision tasks such as segmentation, pose estimation, 3D object detection. However the depth images procured from consumer level senors have non-negligble amounts of noise present in it, which can interfere with the downstream tasks which rely on the depth information to make a decision such as in autonomous driving. The goal of this project is to leverage data driven models such as neural networks to denoise depth images by incorporating the information present about the scene in the RGB image.</p>
<ins>Approach</ins>  
<p align="justify">The proposed approach draws direct inspiration from monocular depth estimation where the depth image is directly estimated from the RGB image. The goal of this project is to fuse the information from the RGB image to denoise the depth image, in a self-supervised manner. The novelty in our approach is that without the use of any additional information in the form of different vantage points [1] or additional stages [5], a noisy depth image should be denoised by the system.  </p>

<ins>Results</ins>  

<p align="center"> <b>Generated Results for NYU Depth Dataset </b> </p>  

![Alt Text](https://github.com/Vasu-Eranki/DenoisingDepthImages/blob/main/Presentation%20Material/nyu_1.png)
![Alt Text](https://github.com/Vasu-Eranki/DenoisingDepthImages/blob/main/Presentation%20Material/nyu_2.png)  

<p align="center> <b>Generated Results for the TransCG Dataset </b></p>  
 
![Alt Text](https://github.com/Vasu-Eranki/DenoisingDepthImages/blob/main/Presentation%20Material/transcg_1.png)
![Alt Text](https://github.com/Vasu-Eranki/DenoisingDepthImages/blob/main/Presentation%20Material/transcg_2.png)  
<p align="center"> Table 1 : Results against two datasets for various algorithms </p>  

 Dataset |Metric| Bilateral Filter | SOTA [1] | MSE w AWGN Noise | MSE | MSE w Group Sparsity  | MSE w downstream tasks
---| --- | --- | --- | ---| ---| ---| ---
NYU Depth Dataset |  MAE |  16.41mm|  44.34mm|  <b>8.58mm</b>|  16.74mm|  11.75mm|  10.01mm| 15.31mm
NYU Depth Dataset |  RMSE | 37.62mm| 196.89mm| 30.15mm| 36.30mm| <b>30.05mm</b>| <b>24.73mm</b>| 34.21mm| 
TransCG Dataset |  MAE |  41.03mm| 49.24mm| <b>11.02mm</b>| 31.01mm| 35.99mm| 16.35mm| 37.81mm| 
TransCG Dataset |  RMSE|  84.90mm| 169.32mm| 37.78mm| 42.12mm|  46.30mm| <b>32.45mm</b>| 49.05mm|  


 <p></br></p> 
<p align="left"> Table 2: Time to process one frame for various algorithms </p>


  
Algorithm |  Inference Time 
---|  ---
Bilateral Filter |  22ms 
Anisotropic Diffusion based Filter |  0.64s
SOTA [1] |  16ms - On a T4 GPU (8GB of RAM)
Proposed Architecture (UNet) |  12.8ms - On a T4 GPU (8GB of RAM)
  
# Team

* Vasu Eranki 

# Required Submissions

* [Proposal](proposal.md)
* [Midterm Checkpoint Presentation Slides](https://docs.google.com/presentation/d/1Kyzuc4vfThnysSqpJy_nGMxYRILOdqt1/edit?usp=sharing&ouid=109510607650224076456&rtpof=true&sd=true)
* [Final Presentation Slides](https://docs.google.com/presentation/d/1Qz5Prh5TxgHiTDqIplJr0FZuXQBvTlMh/edit?usp=sharing&ouid=109510607650224076456&rtpof=true&sd=true)
* [Final Report](report)
