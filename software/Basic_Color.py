import os 
import cv2 
import sys
import json
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt 
from Utilities import * 

depth_scale = 2.5e-4
 
def main():
    parentpath = "C:\\Users\\Vasu\\Desktop\\RGBD\\November_13th\\"
    configs = read_json_file(parentpath)
    colorreadings = "Color Readings\\"
    depthreadings = "Depth Readings\\"
    colorimages = []
    depthimages = []
    colorimages,depthimages = extractdata(parentpath,colorreadings,depthreadings)
    plt.show()
        
    visualise_images(parentpath+colorreadings,parentpath+depthreadings,configs)
    error = processdepthmaps(depthimages,configs)
    MAE,MSE = errorprocess(error,configs)
    #plot_values(MAE,MSE)
    '''
    rangex = [i for i in range(len(MAE))]
    plt.scatter(rangex,MAE,c='r',marker="*")
    plt.plot(rangex,MAE,c='b',marker="*")
    plt.title("Mean Absolute Error")
    plt.ylabel("Error in cm")
    plt.show()
    '''
    makeDF(MAE)
    #makeDF(MSE)
if __name__=="__main__":

    main()
    