# Abstract


<ins>Objective</ins>  
<p align ="justify">Depth maps are a critical part of many computer vision tasks such as segmentation, pose estimation, 3D object detection. However the depth images procured from consumer level senors have non-negligble amounts of noise present in it, which can interfere with the downstream tasks which rely on the depth information to make a decision such as in autonomous driving. The goal of this project is to leverage data driven models such as neural networks to denoise depth images by incorporating the information present about the scene in the RGB image.</p>
<ins>Approach</ins>  
<p align="justify">The proposed approach draws direct inspiration from monocular depth estimation where the depth image is directly estimated from the RGB image. The goal of this project is to fuse the information from the RGB image to denoise the depth image, in a self-supervised manner. The novelty in our approach is that without the use of any additional information in the form of different vantage points [1] or additional stages [5], a noisy depth image should be denoised by the system. </p>

<ins>Results</ins>  

# Team

* Vasu Eranki 

# Required Submissions

* [Proposal](proposal.md)
* [Midterm Checkpoint Presentation Slides](https://docs.google.com/presentation/d/1Kyzuc4vfThnysSqpJy_nGMxYRILOdqt1/edit?usp=sharing&ouid=109510607650224076456&rtpof=true&sd=true)
* [Final Presentation Slides](https://docs.google.com/presentation/d/1Qz5Prh5TxgHiTDqIplJr0FZuXQBvTlMh/edit?usp=sharing&ouid=109510607650224076456&rtpof=true&sd=true)
* [Final Report](report)
