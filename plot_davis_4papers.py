import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
path='E:/code/tracking/SiamMask/SiamMask/experiments/siammask_sharp/test_diffuse5/DAVIS2016/SiamMask'
path_base='E:/code/tracking/SiamMask/SiamMask/experiments/siammask_sharp/test_diffuse/DAVIS2016/SiamMask'
# folders= os.listdir(path)
folders=['blackswan',  'camel', 'car-roundabout', 'car-shadow', 'cows',  'drift-chicane', 'drift-straight', 'goat', 'horsejump-high', 'kite-surf', 'libby', 'motocross-jump', 'paragliding-launch', 'parkour', 'scooter-black', 'soapbox']
print(folders)
visualize_ =1
num_fig=12
num_col = 2
num_row = int(num_fig/num_col)
print(num_row)
# if vis_diffuse:
fig, ax = plt.subplots(num_row, int(num_col*2), figsize=(30, 30))
row_id2=0
from pdb import set_trace
for i in range(num_fig):# for i in range(6):
	folder= folders[i]
	image_file = path+'/'+ folder +"/00020_difmask_show.png"
	im1 = cv2.imread(image_file)
	image_file = path_base+'/'+ folder +"/00020_base.png"
	im2 = cv2.imread(image_file)
	# set_trace()

	# im1= (im1*1.2).astype(np.uint8)
	# row_id2 = obj_id%num_fig
	row_id2= int(i/num_col)
	col_id = int((i%num_col)*2)
	ax[row_id2, col_id].imshow(im2)
	ax[row_id2, col_id].set_title(folder+": baseline")
	ax[row_id2, col_id+1].imshow(im1)
	ax[row_id2, col_id+1].set_title(folder+": SDForest")
	# if img_neg !=0:
	#     ax[1, 2].imshow(img_neg)
for a in ax.ravel():
    a.set_axis_off()
plt.tight_layout()
plt.savefig('davis2016.pdf')  
plt.show()
# return superpix_slic