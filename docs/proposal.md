## Project Proposal

## 1. Motivation & Objective

<p align ="justify">Depth maps are a critical part of many computer vision tasks such as segmentation, pose estimation, 3D object detection, however the depth images procured from consumer level senors have non-negligble amounts of noise present in it, which can interfer the downstream tasks which rely on the depth information to make a decision such as in autonomous driving. The goal of this project is two fold, the first is to leverage the information present in the RGB image to denoise the depth image. The second part of this project is to leverage data driven models like neural networks to denoise the depth image. </p>

## 2. State of the Art & Its Limitations

How is it done today, and what are the limits of current practice?

## 3. Novelty & Rationale

What is new in your approach and why do you think it will be successful?

## 4. Potential Impact

If the project is successful, what difference will it make, both technically and broadly?

## 5. Challenges

What are the challenges and risks?

## 6. Requirements for Success

What skills and resources are necessary to perform the project?

For this project, the following skills are resources are needed for the project to be successful. 

<ins>Skills</ins>  
- Critical Thinking and Analysis
- Knowledge of Python  
    - Knowledge of TensorFlow/PyTorch and NumPy
- Deep Learning Architectures with a focus on Comptuer Vision Architectures  
     - Linear Algebra and Probability for reading deep learning publications  
- Classical Computer Vision Techniques
- Classical Machine Learning Techniques like Naive Bayes


<ins>Resources</ins>
- RGB-D Sensor &#8594; Time of Flight Sensor &#8594; Intel RealSense L515 (LiDAR based)
- CUDA and CuDNN compatible GPU's for accelerating training (For speedy training minimum is an Nvidia A100)
    - Minimum 8 GB of RAM to process data locally
    - Hardware requirements are being met by Google Colab
## 7. Metrics of Success

What are metrics by which you would check for success?

## 8. Execution Plan

Describe the key tasks in executing your project, and in case of team project describe how will you partition the tasks.

Task    | Description 
------| ---
Task 1 | Develop a basic noise model using first principle methods by recording the sensor's noise against varied scenes
Task 2 | Use classical computer vision techniques such as bilateral filter to denoise the depth image
Task 3 | Use basic deep learning based methods such as autoencoders to denoise the depth image
Task 4 | Use pre-trained deep learning based methods to set the baseline 
Task 5 | Develop deep learning based methods that leverages both Depth and RGB information to Denoise the Depth Image

## 9. Related Work

### 9.a. Papers

Paper 1: Self Supervised Deep Depth Denoising [[1]](#1)

Paper 2: High Quality Self-Supervised Deep Image Denoising [[2]](#2)  

Paper 3: Depth image denoising using nuclear norm and learning graph model [[3]](#3)
 
Paper 4: Spatial Hierarchy Aware Residual Pyramid Network for Time-of-Flight Depth Denoising [[4]](#4)

Paper 5: Unsupervised Depth Completion and Denoising for RGB-D Sensors [[5]](#5)
### 9.b. Datasets
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
