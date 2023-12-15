import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from matplotlib import ticker
import cv2
from tqdm import tqdm 
import sklearn 
from sklearn.mixture import GaussianMixture , BayesianGaussianMixture
from scipy import stats 
import scipy as sp
def makeDF(errors):
    color_range = [0,127,255]
    colors = []
    [[[colors.append([color_range[i],color_range[j],color_range[k]]) for k in range(0,len(color_range))] for j in range(0,len(color_range))] for i in range(0,len(color_range))]
    colors = pd.DataFrame(colors)
    colors.rename(columns={0:"R",1:"G",2:"B"},inplace=True)
    errors = pd.DataFrame(errors)

    errors.rename(columns={0:"Error"},inplace=True)
    color_error = pd.concat([colors,errors],axis=1)
    R_error = color_error[["R","Error"]].groupby(["R"]).mean() 
    G_error = color_error[["G","Error"]].groupby(["G"]).mean()
    B_error = color_error[["B","Error"]].groupby(["B"]).mean() 
    print(color_error)
    plt.scatter(R_error.index,R_error["Error"],c='r',marker="*",linewidth=3)
    plt.plot(R_error.index,R_error["Error"],c='r',linestyle="dashed",linewidth=3)
    
    plt.scatter(G_error.index,G_error["Error"],c='g',marker="o",linewidth=3)
    plt.plot(G_error.index,G_error["Error"],c='g',linestyle="dashed",linewidth=3)
    
    plt.scatter(B_error.index,B_error["Error"],c='b',marker="p",linewidth=3)
    plt.plot(B_error.index,B_error["Error"],c='b',linestyle="dashed",linewidth=3)
    plt.title("Mean Absolute Error vs RGB Pixel Values")
    plt.xlabel("Pixel Value [0 or 127 or 255]")
    plt.ylabel("Error (mm)")
    plt.show()
	
def generate_colors(rgb):
    for r in [0,127,255]:
        for g in [0,127,255]:
            for b in [0,127,255]:
                rgb.append((r/255,g/255,b/255))
def plot_errors(errors,rgb):
    for count,i in enumerate(errors): 
            plt.scatter(count,errors[count],color=(rgb[count][0],rgb[count][1],rgb[count][2]))
    plt.plot(errors,c="black",linestyle="dotted")
    plt.title("Mean Absolute Error vs Color")
        
    labels = ["" for i in range(27)]
    labels[0] = "Dark Colors"
    labels[12] = "Warm Colors"
    labels[26]="Bright Colors"
    plt.xticks(np.arange(27),labels,minor=False)
    plt.ylabel("Error(mm)")
    plt.xlabel("Color")
    
    plt.show()
def visualise_errors(original_data,color_data,rgb):
    pbar = tqdm(total = len(original_data))
    errors = original_data - 0.5
    fig,axs = plt.subplots(5,6,figsize=(10,15))
    for count,i in enumerate(errors):
        rows = count//6
        cols = count%6
        axs[rows,cols].imshow(errors[count],cmap='jet')
        #axs[rows,cols].set_ylabel("Flattened Distribution")
        pbar.update(1)
    plt.figure(1)
    pbar = tqdm(total = len(original_data))
    fig,axs = plt.subplots(5,6,figsize=(10,15))
    for count,i in enumerate(errors):
        rows = count//6
        cols = count%6
        axs[rows,cols].imshow(color_data[count])
        #axs[rows,cols].set_ylabel("Flattened Distribution")
        pbar.update(1)
    plt.figure(2)
    pbar = tqdm(total = len(original_data))
    fig,axs = plt.subplots(5,6,figsize=(10,15))
    for count,i in enumerate(errors):
        rows = count//6
        cols = count%6
        axs[rows,cols].hist(errors[count].reshape(-1),density=True,bins=10)
        #axs[rows,cols].set_ylabel("Flattened Distribution")
        pbar.update(1)
    plt.show()
    
def get_bandwidth(samples):
    n = len(samples)
    std = np.std(samples)
    iqr = stats.iqr(samples)/1.34
    bw = min(iqr,std)*0.9*n**(-0.2)
    return bw
def gmm(screen):
    mixture_model = []
    print(screen.shape)
    screen = screen.reshape(-1,1)
    print(screen.shape)
    g_m_m = BayesianGaussianMixture(n_components=3,n_init=1,tol=1e-8,verbose=1,max_iter=100).fit(screen) ## Gaussian - 50, Bayesian - 100
    new_samples,labels = g_m_m.sample(len(screen))
    wasserstein_distance=  stats.wasserstein_distance(screen.reshape(-1),new_samples.reshape(-1))
    print("Wasserstein Distance",wasserstein_distance)
     
    bw = get_bandwidth(new_samples.reshape(-1))
    bw1 = get_bandwidth(screen.reshape(-1))
  
    print("BW",bw*wasserstein_distance,"BW1",bw1*wasserstein_distance)
    
    fig,axs = plt.subplots(1,1)
    generated_hist = np.histogram(1000*new_samples.reshape(-1),bins=100)
    og_hist = np.histogram(1000*screen.reshape(-1),bins=100)
    og_hist_dist = stats.rv_histogram(og_hist,density=False)
    generated_hist_dist = stats.rv_histogram(generated_hist,density=False)
    X = np.linspace(np.min(1000*screen),np.max(1000*screen),10**6)
    fig,ax = plt.subplots()
    ax.plot(X,og_hist_dist.cdf(X),c='r',label="Original CDF")
    ax.plot(X,generated_hist_dist.cdf(X),c='b',label="Generated CDF")
    ax.set_xlabel("Error in (mm)")
    ax.set_ylabel("Estimated Probability")
    ax.set_title("CDF of Generated Distribution and Original Distribution")
    ax.legend()
    plt.figure()
    fig,ax = plt.subplots()
    ax.plot(X,og_hist_dist.pdf(X),c='r',label="Original PDF")
    ax.plot(X,generated_hist_dist.pdf(X),c='b',label="Generated PDF")
    ax.set_xlabel("Error in (mm)")
    ax.set_ylabel("Estimated Probability")
    ax.set_title("PDF of Generated Distribution and Original Distribution")
    ax.legend()
    
    plt.show()

    print("Weights")
    print(g_m_m.weights_)
    print("Means")
    print(g_m_m.means_)
    print("Covariances")
    print(g_m_m.covariances_)
    
        
     
def main():
    screen = np.load("Depth.npy")
    color_screen = np.load("Color.npy")

    screen = screen[:,250:350,300:500]
    color_screen = color_screen[:,250:350,300:500,:].astype(np.uint8)
    errors = [np.mean(i-0.4995)*1000 for i in screen]
    rgb = []
    generate_colors(rgb)
    #plot_errors(errors,rgb)
    #makeDF(errors)
    #visualise_errors(screen,color_screen,rgb)
    gmm(screen-0.5)
    
    
if __name__=="__main__":
    main()