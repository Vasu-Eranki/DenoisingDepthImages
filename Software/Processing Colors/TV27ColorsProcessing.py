import numpy as np 
import pandas as pd 
import cv2 
import matplotlib.pyplot as plt 
import json


def processdepthmaps(depthimages,config):
    error = []
    measurement = config["true_depth"]
    for i in depthimages:
        i = i[config["row_start"]:config["row_end"],config["col_start"]:config["col_end"]] 
        error.append((i-measurement)*100) # Converting m to cm
        
    return error
 
 
def errorprocess(errors,config):
    dark_noise = config["Dark Noise"]
    MAE =  [np.mean(abs(i)) for i in errors]
    MSE = [np.mean(i**2) for i in errors]
    return MAE,MSE
 
     
def makeDF(errors,config):
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
    plt.savefig(config["parentpath"]+"MAE.png")


if __name__=="__main__":