import os 
import cv2 
import sys
import json
import numpy as np 
import pandas as pd 
from PIL import Image
import matplotlib.pyplot as plt 
depth_scale = 2.5e-4

def read_raw_files(path):

    img_data = open(path,'rb').read()
    rows = 640
    cols = 480
    resolution = rows * cols
    img = np.frombuffer(img_data, count=resolution, dtype=np.uint16)
    img = img.reshape((rows, cols))
    return img
    
    
def crop(img,row_start,row_stop,col_start,col_stop):
    return img[row_start:row_stop,col_start:col_stop]
    
    
def visualise_images(colorpaths,depthpaths,configs):
    colorpaths = [colorpaths+i for i in os.listdir(colorpaths)]
    colorpaths_new = []
    depthpaths_new = []
    for i in colorpaths:
        if("Blank" in i):
            colorpaths_new.insert(0,i)
        else:
            colorpaths_new.append(i)
    depthpaths = [depthpaths+i for i in os.listdir(depthpaths) if ".raw" in i]
    for i in depthpaths:
        if("Blank" in i):
            depthpaths_new.insert(0,i)
        else:
            depthpaths_new.append(i)
    colorpaths  = colorpaths_new
    depthpaths = depthpaths_new
    for color_image,depth_image in zip(colorpaths,depthpaths): 
        colorimg = np.asarray(Image.open(color_image))
        depthimg = read_raw_files(depth_image)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depthimg, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = colorimg.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(colorimg, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            resized_color_image = resized_color_image[configs['row_start']:configs['row_end'],configs['col_start']:configs['col_end']]
            depth_colormap = depth_colormap[configs['row_start']:configs['row_end'],configs['col_start']:configs['col_end']]
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)
 
 
def processdepthmaps(depthimages,config):
    error = []
    measurement = config["true_depth"]
    for i in depthimages:
        i = i[config["row_start"]:config["row_end"],config["col_start"]:config["col_end"]] 
        i = i*depth_scale
        error.append((i-measurement)*100) # Converting m to cm
        
    return error
 
 
def errorprocess(errors,config):
    dark_noise = config["Dark Noise"]
    if(dark_noise):
        offset = errors[0]
        offset = 0
        errors = [errors[i]-offset for i in range(0,len(errors)) if i!=1]
    else:
        offset = errors[0]
        offset = 0
        errors = [i-offset for i in errors]
    MAE =  [np.mean(abs(i)) for i in errors]
    MSE = [np.mean(i**2) for i in errors]
    return MAE,MSE
 

 
def read_json_file(path):
    x = [i for i in os.listdir(path) if ".json" in i]
    config = json.load(open(path+x[0]))
    return config
  


  
def extractdata(parentpath,colorreadings,depthreadings):
    colorimages = []
    depthimages = []
    for index,colorimage in enumerate(os.listdir(parentpath+colorreadings)):
        
        image = np.array(Image.open(parentpath+colorreadings+colorimage))
        image = cv2.resize(image, dsize=(480, 640), interpolation=cv2.INTER_AREA)
        if("Blank" in colorimage):
            colorimages.insert(0,image)
        else:
        
            colorimages.append(image)
        
    depthimages_ondisk = [i for i in os.listdir(parentpath+depthreadings) if ".raw" in i]
    
    for index,depthimage in enumerate(depthimages_ondisk):
        depthimage_from_disk = read_raw_files(parentpath+depthreadings+depthimage)
        # plt.imshow(depthimage_from_disk)
        if("Blank" in depthimage):
            depthimages.insert(0,depthimage_from_disk)
        else:
            depthimages.append(depthimage_from_disk)
    

        
    return colorimages,depthimages
    
    
def makeDF(errors):
    color_range = [0,127,255]
    colors = []
    [[[colors.append([color_range[i],color_range[j],color_range[k]]) for k in range(0,len(color_range))] for j in range(0,len(color_range))] for i in range(0,len(color_range))]
    colors = pd.DataFrame(colors)
    colors.rename(columns={0:"R",1:"G",2:"B"},inplace=True)
    errors = pd.DataFrame(errors)

    errors.rename(columns={0:"Error"},inplace=True)
    color_error = pd.concat([colors,errors],axis=1)
    print(color_error)
    R_error = color_error[["R","Error"]].groupby(["R"]).mean() 
    G_error = color_error[["G","Error"]].groupby(["G"]).mean()
    B_error = color_error[["B","Error"]].groupby(["B"]).mean() 
    
    plt.scatter(R_error.index,R_error["Error"],c='r',marker="*")
    plt.plot(R_error.index,R_error["Error"],c='r')
    
    plt.scatter(G_error.index,G_error["Error"],c='g',marker="o")
    plt.plot(G_error.index,G_error["Error"],c='g')
    
    plt.scatter(B_error.index,B_error["Error"],c='b',marker="p")
    plt.plot(B_error.index,B_error["Error"],c='b')
    plt.title("Mean Absolute Error")
    plt.xlabel("Pixel Value [0 or 127 or 255]")
    plt.ylabel("Error (cm)")
    plt.show()
    
    
def plot_values(MAE,MSE):
    rangex = [i for i in range(len(MAE))]
    fig,axs = plt.subplots(1,2)
    axs[0].scatter(rangex,MAE,c='r',marker="*")
    axs[0].plot(rangex,MAE,c='b')
    axs[0].set_title("Mean Absolute Error")
    axs[1].plot(rangex,MSE,c='b')
    axs[1].scatter(rangex,MSE,c='r',marker="*")
    axs[1].set_title("Mean Squared Error")
    plt.show()