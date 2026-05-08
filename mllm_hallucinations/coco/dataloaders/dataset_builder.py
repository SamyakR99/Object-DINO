import os
from .coco14 import CocoDetection

def build_dataset(data_split, annFile=""):
    print(' -------------------- Building Dataset ----------------------')
    root = '/scratch/bcyh/dataset/coco/'
    if annFile != "":
        annFile = os.path.join(root, 'annotations', annFile)
    
    img_size = 448
    
    return CocoDetection(root, 
                        data_split, img_size,
                        annFile=annFile,
                        )