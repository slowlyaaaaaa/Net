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
        # print(self.img_path_all)


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
        img_path_all = [i for i in Path(root_dir).glob('*/*.npy')]
        img_fix = []
        for i in range(len(img_path_all)//2):
            img_fix.append(img_path_all[i])
            img_fix.append(img_path_all[-i])
            print(img_path_all[i])
            print(img_path_all[-i])
        # img_path_all_ = img_path_all.copy()
        # len_ = len(img_path_all_) // 2
        # img_path_all = [img_path_all_[i] if i % 2 == 1 else img_path_all_[-i] for i in range(len(img_path_all_))]
        return img_fix



root_dir = r"C:\Users\yu\Desktop\肺结节分类\分类数据"
# 有肺结节的数据集
train_dataset = MyDataset(root_dir,'train')
# 验证集
val_dataset = MyDataset(root_dir, 'val')



# train_transform = transforms.Compose([
#     transforms.RandomVerticalFlip(),
#     transforms.RandomRotation(degrees=(-30, 30)),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
#     transforms.ToTensor(),
#     normalize])
#
#
# val_transform = transforms.Compose([
#     transforms.ToTensor(),
#     normalize])


# dataloader
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = LeNet5().to(device)

# 定义一个损失函数
loss_fn = nn.CrossEntropyLoss()

# 定义一个优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)#0.01

# 学习率每隔10轮变为原来的0.5
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5) #0.5


# 定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0
    for batch, (x, y) in tqdm(enumerate(dataloader),total=len(dataloader)):
        image, y = x.to(device), y.to(device)
        output = model(image)
        output = output.squeeze(-1).squeeze(-1)
        cur_loss = loss_fn(output, y)
        # cur_loss = f.cross_entropy(output, y)
        _, pred = torch.max(output, axis=1)
        cur_acc = torch.sum(y == pred) / output.shape[0]

        # 反向传播
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1

    train_loss = loss / n
    train_acc = current / n
    print('train_loss=' + str(train_loss))
    print('train_acc=' + str(train_acc))
    return train_loss, train_acc


# 定义一个验证函数
def val(dataloader, model, loss_fn):
    # 将模型转化为验证模型
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch, (x, y) in tqdm(enumerate(dataloader),total=len(dataloader)):
            image, y = x.to(device), y.to(device)
            output = model(image)
            output = output.squeeze(-1).squeeze(-1)
            cur_loss = loss_fn(output, y)
            # cur_loss = f.cross_entropy(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1

    val_loss = loss / n
    val_acc = current / n
    print('val_loss=' + str(val_loss))
    print('val_acc=' + str(val_acc))
    return val_loss, val_acc


# 定义画图函数
def matplot_loss(train_loss, val_loss):
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("训练集和验证集loss值对比图")
    plt.show()


def matplot_acc(train_acc, val_acc):
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.title("训练集和验证集acc值对比图")
    plt.show()


# 开始训练
loss_train = []
acc_train = []
loss_val = []
acc_val = []

epoch =20

min_acc = 0
for t in range(epoch):
    lr_scheduler.step()
    print(f"epoch{t + 1}\n-----------")
    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
    val_loss, val_acc = val(val_dataloader, model, loss_fn)

    loss_train.append(train_loss)
    acc_train.append(train_acc)
    loss_val.append(val_loss)
    acc_val.append(val_acc)

    # 保存最好的模型权重
    if val_acc > min_acc:
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir('save_model')
        min_acc = val_acc
        print(f"save best model, 第{t + 1}轮")
        torch.save(model.state_dict(), 'save_model/best_model.pth')
    # 保存最后一轮的权重文件
    if t == epoch - 1:
        torch.save(model.state_dict(), 'save_model/last_model.pth')

matplot_loss(loss_train, loss_val)
matplot_acc(acc_train, acc_val)
print('Done!')
