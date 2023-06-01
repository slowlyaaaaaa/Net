# 读lbl和img，然后判断lbl的切片中是否全为0，取不全为0的为正样本，然后取对应数量的全为0为负样本，1：1，给他们起名的后边加0或1为标签
# 保存到两个文件夹——预处理
# 读lbl和img
from pathlib import Path
import numpy as np
from random import random,sample
lbl_path = Path(r'C:\Users\yu\Desktop\肺结节分类\有肺结节的\lbl')
img_path = Path(r'C:\Users\yu\Desktop\肺结节分类\有肺结节的\img')

list_lbl = [i for i in lbl_path.glob('*')]
for id_lbl in list_lbl:
    # 读lbl和img
    one_lbl = np.load(str(id_lbl))
    one_img = np.load(str(id_lbl).replace('lbl','img'))
    print(one_lbl.shape)  # z y x
    # 判断lbl的切片中是否全为0，取不全为0的为正样本，然后取对应数量的全为0为负样本
    list_img_1 = []
    list_img_0 = []
    list_lbl_1 = []
    list_lbl_0 = []
    for slice in range(one_lbl.shape[0]):
        one_slice_lbl = one_lbl[slice,:,:]
        one_slice_img = one_img[slice, :, :]
        # print(one_slice_img)
        if np.max(one_slice_lbl) ==1:
            list_img_1.append(one_slice_img)
            list_lbl_1.append(one_slice_lbl)
            # np.save(r'C:\Users\yu\Desktop\肺结节分类\分类数据\1\{}'.format(id_lbl.name+'_1.npy'))
        else:
            list_img_0.append(one_slice_img)
            list_lbl_0.append(one_slice_lbl)
            # np.save(r'C:\Users\yu\Desktop\肺结节分类\分类数据\0\{}'.format(id_lbl.name + '_0.npy'))
        # 我们有了少量正样本的2d切片和大量负样本的2d切片，需要配比为1：1
    # we_need_count = len(list_img_1)*2
    # if len(list_img_0)>len(list_img_1)*2:
    #     list_img_0 = sample(list_img_0, we_need_count)
    for i_1 in range(len(list_img_1)):
        np.save(r'C:\Users\yu\Desktop\肺结节分类\分类数据\1\{}'.format(str(id_lbl.name)+ str(i_1) + '_1.npy'),list_img_1[i_1])
    for i_0 in range(len(list_img_0)):
        np.save(r'C:\Users\yu\Desktop\肺结节分类\分类数据\0\{}'.format(str(id_lbl.name)+ str(i_0) + '_0.npy'),list_img_0[i_0])
    # for i_1 in range(len(list_lbl_1)):
    #     np.save(r'C:\Users\yu\Desktop\肺结节分类\分类数据\1_lbl\{}'.format(str(id_lbl.name)+ str(i_1) + '_1.npy'),list_lbl_1[i_1])
    # for i_0 in range(len(list_lbl_0)):
    #     np.save(r'C:\Users\yu\Desktop\肺结节分类\分类数据\0_lbl\{}'.format(str(id_lbl.name)+ str(i_0) + '_0.npy'),list_lbl_0[i_0])

# path = r'C:\Users\yu\Desktop\肺结节分类\分类数据\0\0_0.npy7_0.npy'
# a = np.load(path)
# import matplotlib.pyplot as plt
# import cv2
# plt.imshow(a)
# plt.show()
