import torch.nn as nn
# import torch.nn as nn.functional


class LeNet5(nn.Module):
    def __init__(self, num_classes=2):
        super(LeNet5, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=6, stride=6)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5,padding=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(32, 2, kernel_size=5, padding=2)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.fc1 = nn.Linear(64 * 22 * 22, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, num_classes)
        self.relu = nn.PReLU()
        self.softmax = nn.Softmax(dim=1)  # bs * channel * size

    def forward(self, x, dropout_rate=0.5):
        x = self.pool1(self.relu(self.conv1(x)))  #1*96*96  →  32*48*48    c*size
        # x = nn.Dropout(p=dropout_rate)(x)
        # x = self.pool2(nn.functional.relu(self.conv2(x)))
        x = self.pool1(self.relu(self.conv2(x)))  # 32*48*48  →  64*24*24
        x = self.pool1(self.relu(self.conv3(x)))  # 64*24*24  →  64*12*12
        x = self.pool1(self.relu(self.conv4(x)))  # 64*12*12  →  32*12*12
        x = self.pool2(self.relu(self.conv5(x)))  # 32*6*6  →  2*1*1
        print(x.shape)
        x = self.softmax(x)
        return x


        # x = nn.Dropout(p=dropout_rate)(x)
        # x = x.view(-1, 64 * 22 * 22)
        # x = nn.functional.relu(self.fc1(x))
        # x = nn.functional.relu(self.fc2(x))
        # x = self.fc3(x)
