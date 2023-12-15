import numpy as np
import cv2
import threading 
from sensor_utils import *




        

def main():
    parentpath = "C:\\Users\\Vasu\\Desktop\\RGBD\\November_25th\\"
    config = configuredevice()
    config["L"]=["Jai Shree Ram"]
    config["row_start"]=270
    config["row_end"]=430
    config["col_start"] = 370
    config["col_end"]=680
    config["true_depth"]=0.495
    config["Dark Noise"]=False
    config["parentpath"]=parentpath
    t1 = threading.Thread(target=generate_colors,args=(config,))
    t2 = threading.Thread(target=captureframes,args=(config,))
    t1.start()
    t2.start()
    t1.join(10000)
    t2.join(10000)
    color_details = np.asarray(config["Color"])
    depth_details = np.asarray(config["Depth"])
    np.save(parentpath+"Color",color_details)
    np.save(parentpath+"Depth",depth_details)
    config["path"] = parentpath
    process_color_depth(config)
    
    
    
if __name__=="__main__":
    main()