import numpy as np
import cv2
import threading 
from Allutils   import *
import json
import pickle
import subprocess

        

def main():
    parentpath = "C:\\Users\\Vasu\\Desktop\\RGBD\\All_Colors\\"
    config = configuredevice()
    config["L"]=["C"]
    config["row_start"]=270
    config["row_end"]=430
    config["col_start"] = 370
    config["col_end"]=680
    config["true_depth"]=0.5185
    config["Dark Noise"]=False
    flag = dict({})
    flag["Success"]=False;
    flag["Random"]="JaiShreeRam"
    with open(parentpath+"hello.pkl",'wb') as f:
        pickle.dump(flag,f)
    t1 = threading.Thread(target=generate_colors,args=(config,))
    t2 = threading.Thread(target=captureframes,args=(config,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    with open(parentpath+"color.pkl",'wb') as f1:
        pickle.dump(config["Color"],f1)
    print("Finished dumping color file")
    with open(parentpath+"depth.pkl",'wb') as f2:
        pickle.dump(config["Depth"],f2)
    print("Finished dumping depth file")
    flag["Success"]=True;
    #process_color_depth(config)
    with open(parentpath+"hello.pkl",'wb') as f3:
        pickle.dump(flag,f3)
    print("Finished dumping new config file")
    
    
    
if __name__=="__main__":
    main()