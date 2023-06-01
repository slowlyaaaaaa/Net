import os
from shutil import copy
import random
import numpy as np

def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


# 获取data文件夹下所有文件夹名（即需要分类的类名）
file_path = 'C:\\Users\\yu\\PycharmProjects\\Net\\data_name'
flower_class = [cla for cla in os.listdir(file_path)]

# 创建 训练集train 文件夹，并由类名在其目录下创建5个子目录
mkfile('data/train')
for cla in flower_class:
    mkfile('data/train/' + cla)

# 创建 验证集val 文件夹，并由类名在其目录下创建子目录
mkfile('data/val')
for cla in flower_class:
    mkfile('data/val/' + cla)

# 划分比例，训练集 : 验证集 = 9 : 1
split_rate = 0.2

# 遍历所有类别的全部npy文件并按比例分成训练集和验证集
for cla in flower_class:
    cla_path = file_path + '/' + cla + '/'  # 某一类别的子目录
    npy_files = os.listdir(cla_path)  # npy_files 列表存储了该目录下所有npy文件的名称
    num = len(npy_files)
    eval_index = random.sample(npy_files, k=int(num * split_rate))  # 从npy_files列表中随机抽取 k 个npy文件名称
    for index, npy_file in enumerate(npy_files):
        # eval_index 中保存验证集val的npy文件名称
        if npy_file in eval_index:
            npy_path = cla_path + npy_file
            new_path = 'data/val/' + cla
            npy_data = np.load(npy_path)  # 读取npy文件的数据
            np.save(os.path.join(new_path, npy_file), npy_data)  # 将读取的数据保存到新路径

        # 其余的npy文件保存在训练集train中
        else:
            npy_path = cla_path + npy_file
            new_path = 'data/train/' + cla
            npy_data = np.load(npy_path)  # 读取npy文件的数据
            np.save(os.path.join(new_path, npy_file), npy_data)  # 将读取的数据保存到新路径
        print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar
    print()

print("processing done!")

