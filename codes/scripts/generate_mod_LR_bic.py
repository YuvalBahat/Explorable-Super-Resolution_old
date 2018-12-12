import os
import sys
import numpy as np
from tqdm import tqdm
import cv2

input_folder = '/home/ybahat/Datasets/DIV2K/DIV2K_train_HR_sub'
save_LR_folder = '/home/ybahat/Datasets/DIV2K/DIV2K_train_HR_sub_bicLRx4'

up_scale = 4
mod_scale = 4

def mod_img(image,modulu):
    if image.ndim<3:
        image = image.reshape(list(image.shape)+[1])
    im_shape = image.shape[:2]
    im_shape -= np.mod(im_shape,modulu)
    return image[:im_shape[0],:im_shape[1],:]

if not os.path.exists(save_LR_folder):
    os.makedirs(save_LR_folder)
    print('mkdir [{:s}] ...'.format(save_LR_folder))
else:
    print('Folder [{:s}] already exists. Exit...'.format(save_LR_folder))
    sys.exit(1)

# img_list = []
# for file_name in os.listdir(input_folder):
#     img_list.append(os.path.join(input_folder,file_name))

progress_bar = tqdm(os.listdir(input_folder))
for file_name in progress_bar:
    cur_im = mod_img(cv2.imread(os.path.join(input_folder,file_name)),mod_scale)
    dsize = tuple((np.array(cur_im.shape[:2])/up_scale).astype(np.int32))
    cur_im = cv2.resize(cur_im,dsize,interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(save_LR_folder,file_name),cur_im)


