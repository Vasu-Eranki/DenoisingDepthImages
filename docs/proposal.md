## Project Proposal


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

## 6. Requirements for Success

For this project, the following skills are resources are needed for the project to be successful. 

<ins>Skills</ins>  
- Critical Thinking and Analysis
- Knowledge of Python  
    - Knowledge of TensorFlow/PyTorch, NumPy and OpenCV
- Deep Learning Architectures with a focus on Comptuer Vision Architectures  
     - Linear Algebra and Probability for reading deep learning publications  
- Classical Computer Vision Techniques
- Classical Machine Learning Techniques like Naive Bayes


<ins>Resources</ins>
- RGB-D Sensor &#8594; Time of Flight Sensor &#8594; Intel RealSense L515 (LiDAR based)
- CUDA and CuDNN compatible GPU's for accelerating training (An Nvidia A100 is recommended for speedy training)
    - Minimum 8 GB of RAM to process data locally
    - Hardware requirements are being met by Google Colab
 
- RGB-D Dataset [Listed Below]("9-b-datasets")
## 7. Metrics of Success

<p align="justify">The authors in papers [1]-[5], used the following metrics in their papers to evaluate the performance of their proposed model against other benchmarks. Similarly for this project, the same metrics will be used to evaluate the performance of our proposed model. The most commonly used metrics to evaluate the robustness of the denoising systems are :</p>  

* RMSE   
* MAE

A successful project would require that the model successfully denoises a depth image by leveraging the information present in the RGB image and should achieve the following values for each metric:  

Metric | Value  
--- | ---
RMSE | 58mm
MAE | 25mm 

## 8. Execution Plan

Describe the key tasks in executing your project

Task| Description 
---| ---
Task 1 | Develop a basic noise model by recording the sensor's output against varied scenes to capture the noise statistics
Task 2 | Use classical computer vision techniques such as bilateral filter to denoise the depth image
Task 3 | Use basic deep learning based methods such as autoencoders to denoise the depth image
Task 4 | Use pre-trained deep learning based methods to set the baseline 
Task 5 | Develop deep learning based methods that leverages both Depth and RGB information to Denoise the Depth Image

## 9. Related Work

### 9.a. Papers

The following papers are of interest and are relevant to the project, since the depth denoisers developed in [[1]](#1) - [[5]](#5) didn't require access to clean-noisy pairs of images. The key idea from each author's paper has been mentioned below.  

Paper 1: Self Supervised Deep Depth Denoising [[1]](#1)  

<p align="justify">&#8594; In this paper, the authors trained a depth denoiser on noisy depth images by exploting the idea that different vantage points will capture the same scene differently. While the goal of this project is to perform monocular depth denoising, the key idea from this paper that seems relevant is that the information from neighbouring pixels can be used to denoise a depth image.</p>

Paper 2: High Quality Self-Supervised Deep Image Denoising [[2]](#2)
<p align = "justify">&#8594; In this paper, the authors develop a denoiser without having access to ground truth data by leveraging CNN's and Bayesian statistics wherein the value of the depth is highly dependent on its neighbours. The paper delves into more details than others, about the noise function which is relevant to this project.</p>  
 
Paper 3: Depth image denoising using nuclear norm and learning graph model [[3]](#3)  

<p align='justify'>&#8594; In this paper, the authors use a Laplacian graph model and convex optimization techniques to reduce the trace norm of the image, to remove the high frequency noise elements inside. The key takeaway from this is that data driven models are not the only way to go about it and convex optimization based techniques are robust enough and have the added benefit of being interpretable when DNN's may not be. </p>

Paper 4: Spatial Hierarchy Aware Residual Pyramid Network for Time-of-Flight Depth Denoising [[4]](#4)

<p align = "justify">&#8594; In this paper, the authors use a CNN based network to extract multiple patchs from the depth image and then proceed to exploit the information present across the multiple scales to denoise the image. The key element from this paper like the others discussed is that neighbouring elements do carry succint information for denoising. </p> 


Paper 5: Unsupervised Depth Completion and Denoising for RGB-D Sensors [[5]](#5)   

<p align="justify">&#8594; In this paper, the authors build a two stage end-to-end system that first completes the depth map before passing it through a depth denoiser. The completed depth map acted as a pseudo ground truth which was used to train the denoiser. While the goal of this project is to perform monocular depth denoising, the key takeaway from this paper is the training procedure where they randomly dropped values in the depth channel to make the estimator more robust.</p>

### <a id ="9-b-datasets">9.b. Datasets</a>
The following datasets will be used for the project either wholly or parts of it, to train the deep learning model. 

Dataset Name | Year of Release | Type of Images | Sensor
---|---|---|---
CLUBS [[6]](#6) | 2019 | Noisy RGB-D Images | Intel RealSense D415, D435 
NYU Depth V2 [[7]](#7) | 2012 | Noisy RGB-D Images | Microsoft Kinect 
DIODE [[8]](#8) | 2019 | Clean RGB-D Images | FARO Focus S350 


### 9.c. Software

For this project, the following software will be used: 
- Python 3.9
- TensorFlow & Keras 
- Intel RealSense SDK for Windows [[9]](#9)

### 9.c.a Hardware
For this project, the following hardware will be used:
- IntelRealSense L515 [[10]](#10)

## 10. References
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


[To Abstract](#index.md)  
[To Project Report](#report.md)  
