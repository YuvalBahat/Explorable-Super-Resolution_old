MODEL_NAME = 'nf160nb10_QF10_LR1e-4'
UPSAMPLING_FACTOR = 30
FPS = 10

import numpy as np
import os
import cv2
import shutil
root_folder = '/home/tiras/ybahat' if 'tiras' in os.getcwd() else '/media/ybahat/data/projects'
model_dir = os.path.join(root_folder,'SRGAN/experiments',MODEL_NAME)
err_arr = np.load(os.path.join(model_dir,'avg_estimated_err.npz'))['avg_estimated_err']
frame_nums = np.load(os.path.join(model_dir,'avg_estimated_err.npz'))['avg_estimated_err_step']
frames_dir = os.path.join(model_dir,'temp_frames_dir')
os.mkdir(frames_dir)
max_val = np.max(err_arr)

for frame_num in range(err_arr.shape[2]):
    cur_frame = 255*np.expand_dims(np.expand_dims(err_arr[:,:,frame_num],1),-1)/max_val
    cur_frame = cur_frame.repeat(UPSAMPLING_FACTOR,axis=1).repeat(UPSAMPLING_FACTOR,axis=3).reshape([8*UPSAMPLING_FACTOR,8*UPSAMPLING_FACTOR])
    cv2.putText(cur_frame, str(frame_nums[frame_num]), (20, 200), cv2.FONT_HERSHEY_PLAIN, fontScale=2.0,
                color=255, thickness=1)
    cv2.imwrite(os.path.join(frames_dir,'frame%d.png'%(frame_num)),cur_frame)

os.system('gifski --fps %d -o %s/quantization_errors.gif %s/frame*.png'%(FPS,model_dir,frames_dir))
shutil.rmtree(frames_dir)
