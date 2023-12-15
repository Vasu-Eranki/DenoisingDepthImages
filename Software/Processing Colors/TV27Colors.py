import numpy as np
import cv2
import threading 
from sensor_utils import *




        

def main():
    parentpath = "C:\\Users\\Vasu\\Desktop\\RGBD\\November_17th\\3m"
    config = configuredevice()
    config["L"]=["Jai Shree Ram"]
    config["Dark Noise"]=False
    config["parentpath"]=parentpath
    t1 = threading.Thread(target=generate_colors,args=(config,))
    t2 = threading.Thread(target=captureframes,args=(config,))
    t1.start()
    t2.start()
    t1.join(100)
    t2.join(100)
    color_details = np.asarray(config["Color"])
    depth_details = np.asarray(config["Depth"])
    np.save(parentpath+"Color",color_details)
    np.save(parentpath+"Depth",depth_details)
    config["path"] = parentpath
    #process_color_depth(config)
    
    
    
if __name__=="__main__":
    main()