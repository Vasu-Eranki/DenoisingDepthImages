# Table of Contents
* [Abstract](#Abstract)
* [Introduction](#1-introduction)
* [Related Work](#2-related-work)
* [Technical Approach](#3-technical-approach)
* [Evaluation and Results](#4-evaluation-and-results)
* [Discussion and Conclusions](#5-discussion-and-conclusions)
* [References](#6-references)

# Abstract

Depth maps are a critical part of many computer vision tasks such as segmentation, pose estimation, 3D object detection. However the depth images procured from consumer level senors have non-negligble amounts of noise present in it, which can interfere with the downstream tasks which rely on the depth information to make a decision such as in autonomous driving. The goal of this project is to leverage data driven models such as neural networks to denoise depth images by incorporating the information present about the scene in the RGB image.

Depth maps are a critical part of many computer vision tasks such as segmentation, pose estimation, 3D object detection. However, the depth images generated from commerical sensors suffer from multiple sources of noise which can degrade the confidence levels of estimators used in downstream tasks because of the quality of images generated. 

# 1. Introduction

## <ins>I. Acronyms</ins>

Acronym| Full Form
---| ---
SOTA| State-Of-The-Art
MAE | Mean Absolute Error 
RMSE | Root Mean Squared Error 
RGB-D | RGB - Depth Image
TOF | Time of Flight
DNN | Deep Neural Network

## 1. Motivation & Objective

<p align ="justify">Depth maps are a critical part of many computer vision tasks such as segmentation, pose estimation, 3D object detection. However the depth images procured from consumer level senors has non-negligble amounts of noise present in it, which can interfere with the downstream tasks that rely on the depth information to make a decision such as in autonomous driving. The goal of this project is to leverage data driven models such as neural networks to denoise depth images by incorporating the information present about the scene in the RGB image.</p>

## 2. State of the Art & Its Limitations

## 3. Novelty & Rationale

<ins>Approach</ins>
<p align="justify">The proposed approach draws direct inspiration from monocular depth estimation where the depth image is directly estimated from the RGB image. The goal of this project is to fuse the information from the RGB image to denoise the depth image, in a self-supervised manner. The novelty in our approach is that without the use of any additional information in the form of different vantage points [1] or additional stages [5], a noisy depth image should be denoised by the system. </p>  

<ins>Rationale</ins>  

<p align="justify">TOF sensors rely on the delay between emitting and reflection of a light signal to calculate the delay. An object further away from a sensor will have a longer delay and an object closer to the sensor will have a shorter delay. The texture of the object and the color of the object also affect the reflected light and the goal of this project is to focus on the latter. Object colored white will reflects more light when compared to other colors, and objects colored black will reflect very little light when compared to others. The color of the object matters in the amount of light reflected back to the TOF sensor and therefore the amount of noise present in each pixel has a direct relation with the associated color of that pixel, therefore theoretically it should be possible to leverage the color and scene information present in the RGB image to denoise a depth image because of the correlation between color and reflected light.</p>  

<p align="justify"> Furthermore, it is possible to build a basic noise model from observing the noise response to three colors which are red, green and blue since all colors can be constructed as a combination of the three. Therefore the rationale of this project relies on basic color theory that all colors can be constructed from three colors and that colors influence the amount of light reflected back to the sensor which in turn can change the amount of noise introduced to the depth information captured for each pixel.</p>  

## 4. Potential Impact

<p align="justify"> The goal of this project is to show that monocular depth denoising is possible, which means that the input to the model is a noisy RGB-D image and the output is a clean RGB-D image. The ramifications would be a model that is faster and easier to train, since it wouldn't require a larger end-to-end system or multiple sensors capturing the same scene. From an inference perspective, monocular depth denoising would help process frames closer to real-time system requirements since the number of frames that can be processed in a second would be comparatively more than a system which is more complex or requires fusing multiple frames. The  broader perspective is that moncocular depth estimation, could help in downstream tasks like robotics and autonomous driving which require clean depth images in real-time to make sound decisions. </p>  

## 5. Challenges
 
<ins>Challenges</ins>
- Training a DNN based denoiser without ground truth images, will be a challenge. Without the ground truth, there is no simple way to evaluate the amount of noise the denoiser removes from the image since there is no prior to condition it on. 
- Ensuring that sample bias doesn't happen when collecting data to create a simple noise function. Sample bias can show up in two forms:
     - The noise model only works for the specific sensor for which it was collected
     - The noise model only works for specific environment conditions (If surfaces are sleek, ambient light intensity is less than a specific value)       

<ins>Risks</ins>
- The correlation between color and noise is weak or none when it was assumed that there was a strong correlation between the two
- The proposed model only works for specific sensors and is not model agnostic
- The proposed model is less robust to removing noise than other more complex models proposed in [[1]](#1)-[[5]](#5)

# 2. Related Work


<p align="justify">Currently, there are two school of thought on how to tackle this problem, a supervised approach and a self-supervised approach. Because for this project, the goal is to denoise the image without having access to the ground truth, more focus will be on the self-supervised based state of the art model.</p>    

<p align = "justify">There are two neural network models which were proposed in 2019 and 2022 respectively which are the state of the art in depth denoising. The first paper proposed by Sterzentsenko et.al [1] captures the same scene by four different sensors and then uses the fact that the noise from each sensor will be slightly different because of the difference in vantage points and uses this information to denoise the final image. They were able to achieve a MAE of 25.11mm, on a custom dataset that wasn't released to the public. The limitations of this paper are that during training it requires 4 different depth images taken from four different vantage points to denoise the image.</p>  

<p align="justify"> The second paper proposed by Fan et.al in 2022 [5] uses depth estimation as a prior to denoising, and the authors were able to achieve a MAE of 32.8mm on the ScanNet Dataset [11]. The neural network model is trained end to end. The first stage is a depth estimator, and the second stage is a depth denoiser. The imputed depth image is then fed to the depth denoiser which predicts the residual between the original image and the imputed image. During inference, the residual is added to the original noisy depth image to render a denoised depth image. The limiation of this model is similar to the last one, which is that it requires training a depth estimator as well to achieve the task of depth denoising, this requires copious amounts of training data and GPU resources. </p>   


# 3. Technical Approach

# 4. Evaluation and Results

 Dataset |Metric| Bilateral Filter | SOTA [1] | MSE w AWGN Noise | MSE | MSE w Group Sparsity  | MSE w downstream tasks
---| --- | --- | --- | ---| ---| ---| ---
NYU Depth Dataset |  MAE |  16.41mm|  44.34mm|  <b>8.58mm</b>|  16.74mm|  11.75mm|  10.01mm| 15.31mm
NYU Depth Dataset |  RMSE | 37.62mm| 196.89mm| 30.15mm| 36.30mm| <b>30.05mm</b>| <b>24.73mm</b>| 34.21mm| 
TransCG Dataset |  MAE |  41.03mm| 49.24mm| <b>11.02mm</b>| 31.01mm| 35.99mm| 16.35mm| 37.81mm| 
TransCG Dataset |  RMSE|  84.90mm| 169.32mm| 37.78mm| 42.12mm|  46.30mm| <b>32.45mm</b>| 49.05mm| 

Algorithm |  Inference Time 
---|  ---
Bilateral Filter |  22ms 
Anisotropic Diffusion based Filter |  0.64s
SOTA [1] |  16ms - On a T4 GPU (8GB of RAM)
Proposed Architecture (UNet) |  12.8ms - On a T4 GPU (8GB of RAM)



# 5. Discussion and Conclusions

# 6. References
<a id="1">[1]</a>
Sterzentsenko, V., Saroglou, L., Chatzitofis, A., Thermos, S., Zioulis, N., Doumanoglou, A., Zarpalas, D. and Daras, P., 2019. Self-supervised deep depth denoising. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 1242-1251).   URL: www.openaccess.thecvf.com/content_ICCV_2019/papers/Sterzentsenko_Self-Supervised_Deep_Depth_Denoising_ICCV_2019_paper.pdf  

<a id="2">[2]</a>
Laine, S., Karras, T., Lehtinen, J. and Aila, T., 2019. High-quality self-supervised deep image denoising. Advances in Neural Information Processing Systems, 32.  URL: https://proceedings.neurips.cc/paper/2019/file/2119b8d43eafcf353e07d7cb5554170b-Paper.pdf  
<a id ="3">[3]</a>
Yan, C., Li, Z., Zhang, Y., Liu, Y., Ji, X. and Zhang, Y., 2020. Depth image denoising using nuclear norm and learning graph model. ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM), 16(4), pp.1-17.  URL: https://dl.acm.org/doi/pdf/10.1145/3404374

<a id = "4">[4]</a>
Dong, G., Zhang, Y. and Xiong, Z., 2020. Spatial hierarchy aware residual pyramid network for time-of-flight depth denoising. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXIV 16 (pp. 35-50). Springer International Publishing. URL: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690035.pdf  

<a id="5">[5]</a>
Fan, L., Li, Y., Jiang, C. and Wu, Y., 2022, May. Unsupervised Depth Completion and Denoising for RGB-D Sensors. In 2022 International Conference on Robotics and Automation (ICRA) (pp. 8734-8740). IEEE. URL: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9812392

<a id = "6">[6]</a>
Novkovic, T., Furrer, F., Panjek, M., Grinvald, M., Siegwart, R. and Nieto, J., 2019. CLUBS: An RGB-D dataset with cluttered box scenes containing household objects. The International Journal of Robotics Research, 38(14), pp.1538-1548.URL: https://clubs.github.io/  

<a id = "7">[7]</a>
Silberman, N., Hoiem, D., Kohli, P. and Fergus, R., 2012. Indoor segmentation and support inference from rgbd images. In Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part V 12 (pp. 746-760). Springer Berlin Heidelberg. URL: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html  

<a id = "8">[8]</a>
Vasiljevic, I., Kolkin, N., Zhang, S., Luo, R., Wang, H., Dai, F.Z., Daniele, A.F., Mostajabi, M., Basart, S., Walter, M.R. and Shakhnarovich, G., 2019. Diode: A dense indoor and outdoor depth dataset. arXiv preprint arXiv:1908.00463.URL:https://diode-dataset.org/  

<a id = "9">[9]</a>
URL: https://www.intelrealsense.com/sdk-2/  

<a id="10">[10]</a>
URL:https://dev.intelrealsense.com/docs/lidar-camera-l515-datasheet

<a id = "11">[11]</a>
Dai, A., Chang, A.X., Savva, M., Halber, M., Funkhouser, T. and Nießner, M., 2017. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 5828-5839).URL:http://www.scan-net.org/  
