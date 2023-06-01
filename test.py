import os
import torch
from torch import nn
from torch.autograd import Variable

from net import LeNet5
import numpy as np
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from locale import normalize

import torch
from torch import nn
from torchvision.transforms import transforms

from net import LeNet5
import numpy as np
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn.functional as f
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class MyDataset(Dataset):
    def __init__(self, root_dir,str_):
        self.root_dir = root_dir
        self.label_dict = {'without': 0, 'with': 1}  # 标签映射为数值
        self.img_path_all = self.img_path_get(self.root_dir)
        if str_ == 'train':
            self.img_path = self.img_path_all[:len(self.img_path_all)//5*3]
        elif str_ == 'val':
            self.img_path = self.img_path_all[len(self.img_path_all) // 5 * 3:len(self.img_path_all) // 5 * 4]
        else:
            self.img_path = self.img_path_all[len(self.img_path_all) // 5 * 4:]
        # def __getitem__(self, idx):
        #     img_name = self.img_path[idx]
        #     img_itm_path = os.path.join(self.root_dir, self.label_dir, img_name)
        #     img = np.load(img_itm_path)
        #     label = self.label_dict[self.label_dir]  # 将标签名映射为数值
        #     return torch.from_numpy(img), torch.tensor(label)  # 转换为张量类型

    def __getitem__(self, idx):
        # 读lbl和img，然后判断lbl的切片中是否全为0，取不全为0的为正样本，然后取对应数量的全为0为负样本，1：1，给他们起名的后边加0或1为标签
        # 保存到两个文件夹——预处理
        # dataset直接读图，直接取图名的最后一个字符转为int就是分类标签
        img_name = self.img_path[idx]
        img = np.load(str(img_name))
        # img_tensor = torch.from_numpy(img)
        img_tensor = np.expand_dims(img,axis=0)
        img_tensor = torch.tensor(img_tensor,dtype=torch.float32)  # 转换为张量类型
        label_tensor = torch.tensor(int(img_name.stem[-1]), dtype=torch.long)  # 转换为张量类型，标签需要是 long 类型
        return img_tensor, label_tensor

    def __len__(self):
        return len(self.img_path)
    @staticmethod
    def img_path_get(root_dir):
        img_path = [i for i in Path(root_dir).glob('*/*.npy')]
        return img_path


root_dir = r"C:\Users\yu\Desktop\肺结节分类\分类数据"
# 有肺结节的数据集
test_dataset = MyDataset(root_dir,'test')

# dataloader
train_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = LeNet5().to(device)

model.load_state_dict(torch.load(r"C:/Users/yu/PycharmProjects/Net/save_model/best_model.pth"))
classes = [
    "with_lung_nodule", "without_lung_nodule"]

model.eval()
with torch.no_grad():
    for i in tqdm(range(50),total=50):
        x, y = test_dataset[i][0], test_dataset[i][1]
        # plt.imshow(x[0])  # 只显示第一张图片
        # plt.show()
        x = torch.unsqueeze(x, dim=0).float().to(device)
        pred = model(x)
        predicted, actual = classes[torch.argmax(pred[0])], classes[y]
        print(f'predicted:"{predicted}", Actual:"{actual}"')
