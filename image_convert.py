import cv2
import numpy as np
from PIL import Image

import os

def main(im_name):
    def_path=os.path.join(os.path.dirname(__file__),"imputs","node_tf")
    im_n=str(im_name)+".png"
    im_path=os.path.join(def_path,im_n)
    im =Image.open(im_path).convert('L')
    np_im=np.array(im)
    np_im=np_im > 128
    print(np_im)
    out_put_dir=str(im_name)+".txt"
    out_put_dir=os.path.join(def_path,out_put_dir)
    np.savetxt(out_put_dir,np_im,fmt='%.2f',delimiter='\t')

if __name__=="__main__":
    im_name="002"
    main(im_name)