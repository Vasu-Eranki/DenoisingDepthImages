# Table of Contents
* [Abstract](#Abstract)
* [Introduction](#1-introduction)
* [Related Work](#2-related-work)
* [Technical Approach](#3-technical-approach)
* [Evaluation and Results](#4-evaluation-and-results)
* [Discussion and Conclusions](#5-discussion-and-conclusions)
* [References](#6-references)

# Abstract

Provide a brief overview of the project objhectives, approach, and results.

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

<p align="justify">Currently, there are two school of thought on how to tackle this problem, a supervised approach and a self-supervised approach. Because for this project, the goal is to denoise the image without having access to the ground truth, more focus will be on the self-supervised based state of the art model.</p>    

<p align = "justify">There are two neural network models which were proposed in 2019 and 2022 respectively which are the state of the art in depth denoising. The first paper proposed by Sterzentsenko et.al [1] captures the same scene by four different sensors and then uses the fact that the noise from each sensor will be slightly different because of the difference in vantage points and uses this information to denoise the final image. They were able to achieve a MAE of 25.11mm, on a custom dataset that wasn't released to the public. The limitations of this paper are that during training it requires 4 different depth images taken from four different vantage points to denoise the image.</p>  

<p align="justify"> The second paper proposed by Fan et.al in 2022 [5] uses depth estimation as a prior to denoising, and the authors were able to achieve a MAE of 32.8mm on the ScanNet Dataset [11]. The neural network model is trained end to end. The first stage is a depth estimator, and the second stage is a depth denoiser. The imputed depth image is then fed to the depth denoiser which predicts the residual between the original image and the imputed image. During inference, the residual is added to the original noisy depth image to render a denoised depth image. The limiation of this model is similar to the last one, which is that it requires training a depth estimator as well to achieve the task of depth denoising, this requires copious amounts of training data and GPU resources. </p>   

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

# 3. Technical Approach

# 4. Evaluation and Results

# 5. Discussion and Conclusions

# 6. References
