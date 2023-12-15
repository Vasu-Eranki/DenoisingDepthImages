import pyrealsense2 as rs
import numpy as np
import cv2
import time
import threading 
from tqdm import tqdm
import pandas as pd 
import matplotlib.pyplot as plt 

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
    
    
    
    

#---------------------------------------------------------------------------------------#




def configuredevice():
    pipeline = rs.pipeline()
    config = rs.config()
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("Connect a Intel Depth Camera with a Color Sensor")
        exit(0)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)
    align_to = rs.stream.color
    align = rs.align(align_to)
    configs = {"pipeline":pipeline,"align":align,"Depth Scale":depth_scale}
    return configs
   
def turnoffcamera(RGB_counter,config):
    if(RGB_counter%1000==0):
        pipeline = config["pipeline"]
        pipeline.stop()
        time.sleep(60)
        n_config = configuredevice()
        config["pipeline"]= n_config["pipeline"]
        config["align"] = n_config["align"]
        config["Depth Scale"] = n_config["Depth Scale"]
    
def captureframes(config):
    depth_scale = config["Depth Scale"]
    L = config["L"]
    depth_images_across_colors = []
    color_images_across_colors = []
    count = 0
    acquisition_flag = False
    RGB_counter = 0;
    z = input("Would you like to start the experiment ?")
    pbar = tqdm(total=18*18*18)
    frame_stabiliser = 2
    if(z=="Yes"):
        while(RGB_counter<18*18*18):
            zerocolor_image = []
            zerodepth_image = []
            acquisition_flag= False; 
            count = 0
            turnoffcamera(RGB_counter,config)
            pipeline = config["pipeline"]
            align = config["align"]
            while count<=(frame_stabiliser+5):
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not aligned_depth_frame or not color_frame:
                    continue
                    
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data()).astype(np.uint8)
                color_image = cv2.resize(color_image, depth_image.shape[::-1], interpolation = cv2.INTER_AREA)
                true_depth_image =  depth_image*depth_scale
                count+=1; 
                if(count==frame_stabiliser and acquisition_flag==False):
                    acquisition_flag = True
                if(acquisition_flag):
                    zerocolor_image.append(color_image)
                    zerodepth_image.append(true_depth_image)
            color_images_across_colors.append(np.mean(zerocolor_image,axis=0).astype(np.uint8))## 1x640x480x3
            depth_images_across_colors.append(np.mean(zerodepth_image,axis=0).astype(np.float16)) ## 1x640x480
            L[0]="X"
            while(True):
                if(L[0]=="C"):
                    break
            RGB_counter+=1
            pbar.update(1)
    else:
        print("Execution terminated")
    pbar.close()
    L[0]="D"
    config["Color"]=color_images_across_colors
    config["Depth"]=depth_images_across_colors
    pipeline.stop()
def redo_colors(R,G,B):
	B+=15
	if(B//256==1):
		B = 0
		G+=15
		if(G//256==1):
			G = 0
			R+=15
	return R,G,B
    
def repaint_scene(R,G,B):
    image=  np.zeros((1080,1920,3)).astype(np.uint8)
    image[:,:,:]=[R,G,B]
    image= cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    cv2.namedWindow("Color",cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Color",image)
    key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
    
def generate_colors(config):
    L = config["L"]
    R=G=B = 0
    while True:
        data = L[0]
        if(data=="D"):
            return
        if(data=="X"):
            R,G,B = redo_colors(R,G,B)
            repaint_scene(R,G,B)
            L[0]="C"
        repaint_scene(R,G,B)
    
def process_color_depth(config):
    parentpath = config["path"]
    colors = np.load(parentpath+"Color.npy")
    depths = np.load(parentpath+"Depth.npy")
    
    error = processdepthmaps(depths,config)
    MAE,MSE = errorprocess(error,config)
    makeDF(MAE,config) 
