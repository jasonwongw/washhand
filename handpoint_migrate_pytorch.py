import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"  ##表示使用GPU编号为0的GPU进行计算
import time
import copy           # copy模块，用于后面模型参数的深拷贝  copy.deepcopy
import os.path
import json
import matplotlib
import cv2
import pylab
import pandas as pd
#import seaborn as sns
from PIL import Image
from math import sin, cos, pi
import matplotlib.pyplot as plt
import numpy as np
#from tqdm import tqdm
from typing import Optional, Union, Callable, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision
from torchvision import models
from torch.utils.data import random_split
import torch.utils.data as Data
from torchvision import transforms
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

plt.rcParams['figure.figsize'] = (20,20)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

class ConvNormActivation(torch.nn.Sequential):  #修改mobilenet_v2第一层
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU6,
        dilation: int = 1,
        inplace: bool = True,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers = [torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                  dilation=dilation, groups=groups, bias=norm_layer is None)]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer(inplace=inplace))
        super().__init__(*layers)
        self.out_channels = out_channels

class structure:
    def __init__(self,NUM_EPOCHS,BATCH_SIZE,outnode,innode):
        self.epochs=NUM_EPOCHS
        self.batchsize=BATCH_SIZE
        self.outnode=outnode
        self.innode=innode
        self.model = models.mobilenet_v2(pretrained=True, progress=True)  #迁移学习,最好全部层都微调,不要只微调最后的输出层
        for p in self.model.parameters():
            p.requires_grad = True  # 里面所有参数都是不可训练的 ,底部的卷积层全部冻结，在目标域仅仅对顶部的全连接层进行训练
        # 我们只需将最后的线性层的out_features置换成我们分类的类别即可。可将最后的linear层替换成我们自己的linear层，我们自己的
        # linear层的requires_grad默认是True。
        self.model.features[0]=ConvNormActivation(self.innode, 32, kernel_size=3, stride=2,)  #改变输入层的输入通道1
        num_ftrs = self.model.last_channel
        self.model.classifier = nn.Sequential(nn.Dropout(0.2),nn.Linear(num_ftrs,self.outnode))  #改变输出层
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.type(torch.FloatTensor)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005, weight_decay=0.00005)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5, 10, 25, 40, 65], gamma=0.1)
        self.loss_fn = torch.nn.MSELoss()

    def data(self,train_images,train_keypoints,w,h):  #数据转为pytorch格式
        train_images = train_images.astype(np.float32)  # torch.cuda.DoubleTensor->torch.cuda.FloatTensor
        train_keypoints = train_keypoints.astype(np.float32)
        train_x = torch.from_numpy(train_images.reshape(-1, 1, w, h))  # pd->torch tensor and np->torch
        train_y = torch.from_numpy(train_keypoints)  # pd比np多一个.values

        dataset = Data.TensorDataset(train_x, train_y)  # 把训练集和标签继续封装
        self.train_data, self.eval_data = random_split(dataset, [round(0.8 * train_x.shape[0]), round(0.2 * train_x.shape[0])],
                                             generator=torch.Generator().manual_seed(42))  # 划分训练和验证
        print('训练集长度{}'.format(len(self.train_data)))
        traindata = Data.DataLoader(dataset=self.train_data, batch_size=32, shuffle=True, num_workers=0,
                                    drop_last=True)  # drop_last决定不足一个batch_size的数据是否保留
        evaldata = Data.DataLoader(dataset=self.eval_data, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
        print('数据转化为Pytorch格式')
        return traindata,evaldata

    def train(self,traindata,evaldata):
        print('开始训练')
        BATCH_SIZE=self.batchsize
        NUM_EPOCHS=self.epochs
        best_model = None
        min_val_loss = np.inf

        train_steps = len(self.train_data) / BATCH_SIZE
        val_steps = len(self.eval_data) / BATCH_SIZE
        for epoch in range(NUM_EPOCHS):
            torch.manual_seed(1 + epoch)
            print(f"EPOCH: {epoch + 1}/{NUM_EPOCHS}")
            self.model.train()
            train_loss = 0.0
            for inputs, labels in traindata:
                x = inputs
                y = labels
                (x, y) = (x.to(self.device), y.to(self.device))
                if hasattr(torch.cuda, 'empty_cache'):  #判断对象是否包含对应的属性
                    torch.cuda.empty_cache()
                pred = self.model(x)  #这里内存不够
                loss = self.loss_fn(pred, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss

            with torch.no_grad():
                self.model.eval()
                val_loss = 0.0
                for inputs, labels in evaldata:
                    x = inputs
                    y = labels
                    (x, y) = (x.to(self.device), y.to(self.device))
                    pred = self.model(x)
                    val_loss += self.loss_fn(pred, y)

            self.scheduler.step()

            avg_train_loss = train_loss / train_steps
            avg_val_loss = val_loss / val_steps

            print(f"Average train loss: {avg_train_loss:.6f}, Average validation loss: {avg_val_loss:.6f}")

            if min_val_loss > val_loss:
                min_val_loss = val_loss
                best_model = copy.deepcopy(self.model)
            if epoch%20==0: #20轮保存一次
                torch.save(best_model, 'handbox{}.pth'.format(epoch))
        torch.save(best_model, 'my_model_pytorch.pth')
        print('训练完成')

    def predict(self,input, model, device):
        model.to(device)
        with torch.no_grad():
            input=input.to(device)
            out = model(input)
            #_, pre = torch.max(out.data, 1) #分类任务，输出最大值
        return out#pre.item()

    def predict_show(self,mod,inp):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model=torch.load(mod)
        input=inp
        test_img = np.array(Image.open(input))  # 此处得到的是pillow图像Image实例
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        test_img=cv2.resize(test_img,(96,96))
        train_images = test_img.astype(np.float32)  # torch.cuda.DoubleTensor->torch.cuda.FloatTensor
        train_x=torch.from_numpy(train_images.reshape(-1,1,96,96)) #pd->torch tensor and np->torch
        pred=self.predict(train_x,model,device)
        print(pred)
        pred= pred.cpu().numpy() #tensor->numpy(在cpu下转,numpy是cpu-only)
        pred=pred.reshape(-1,2)
        plt.clf()
        test_img1 = np.array(Image.open(input))/255  # 此处得到的是pillow图像Image实例
        test_img1=cv2.resize(test_img1,(96,96)) #?,?->96,96
        for p in range(pred.shape[0]):
            plt.plot(pred[p, 0], pred[p, 1], 'r.')
        plt.imshow(test_img1)
        pylab.show()


# pathx = 'C:/Users/Administrator/Desktop/fsdownload/pe.csv'
# #pathy = '/tmp/pycharm_project_110/train_Y3.csv'
# train_data = pd.read_csv(pathx)
#train_data_Y = pd.read_csv(pathy)

# def load_images(image_data):  #csv格式，图片平铺，多列(n,m)
#     images = []
#     for idx, sample in image_data.iterrows():  # iterrows():pd格式
#         image = np.array(sample)
#         # image = np.array(sample['Image'].split(' '), dtype=int)
#         #image = np.reshape(image, (96, 96, 1))
#         images.append(image)
#     images = np.array(images)
#     return images


# def load_keypoints(keypoint_data):  #关键点
#     keypoint_data = train_data_Y
#     # keypoint_data = keypoint_data.drop('Image',axis = 1)
#     keypoint_features = []
#     for idx, sample_keypoints in keypoint_data.iterrows():
#         keypoint_features.append(sample_keypoints)
#     keypoint_features = np.array(keypoint_features, dtype='float')
#     return keypoint_features

# if __name__ == '__main__':
#     res=structure()


'''可视化网络结构'''
# dummy_input = torch.rand(13, 1, 96, 96)
# with SummaryWriter('runs/exp-1') as w:
#     w.add_graph(model, (dummy_input,))


'''image数据集-pytorch数据集(关键点)'''
# def read_image(path): #path='C:/Users/Administrator/Desktop/yolov5-master/washhand/handpoint_labels/synth1/'
#     '''读取文件夹中的所有图片'''
#     train_x = []
#     train_y = []
#     test_x = []
#     test_y = []
#     files_point = sorted([f for f in os.listdir(path) if f.endswith('.json')])  # os.listdir获取指定文件夹下的所有文件
#
#     #关键点,图片
#     for f in files_point: #读取一张图片手坐标(三维坐标点)
#         with open(path+f, 'r') as fid:
#             dat = json.load(fid)  #json文件中读取数据{'hand_points':...,'is_left':...)
#         pts = np.array(dat['hand_pts'])
#         train_y.append(pts)
#         path_img=path + f[0:-5] + '.jpg'
#         img = cv2.imread(path_img, cv2.IMREAD_COLOR)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         # img = cv2.resize(img, (640, 640))
#         img = np.reshape(img, (img.shape[0], img.shape[1], 1))
#         img = img.transpose(2, 0, 1)
#         img = img / 255  # 归一化
#         train_x.append(img)
#
#     # 转成pytorch数据
#     train_x = torch.tensor(train_x)
#     train_y = torch.tensor(train_y)
#     dataset = Data.TensorDataset(train_x, train_y)  # 把训练集和标签继续封装
#     train_data, eval_data = random_split(dataset, [round(0.8 * train_x.shape[0]), round(0.2 * train_x.shape[0])],
#                                          generator=torch.Generator().manual_seed(42))  # 划分训练和验证
#     traindata = Data.DataLoader(dataset=train_data, batch_size=32, shuffle=True, num_workers=0,drop_last=True)  # drop_last决定不足一个batch_size的数据是否保留
#     evaldata = Data.DataLoader(dataset=eval_data, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
#     return traindata, evaldata

'''image数据集-pytorch数据集(分类)'''
# def read_image(path):
#     '''读取路径下所有子文件夹中的图片'''
#     train_x = []
#     train_y = []
#     test_x = []
#     test_y = []
#     n_class = 0
#     perClassNum = 120   # 每类图片数量
#     for child_dir in os.listdir(path):  # 类
#         child_path = os.path.join(path, child_dir)
#         print(child_path)
#         imgCount = 0
#         testCount = 0
#         for dir_image in tqdm(os.listdir(child_path)):  # 图片读取
#             imgCount += 1
#             if imgCount > perClassNum: # 每类用100张
#                 break
#             img = cv2.imread(child_path + "\\" + dir_image, cv2.IMREAD_COLOR)
#             img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#             img = cv2.resize(img, (640, 640))
#             img = np.reshape(img, (640, 640, 1))
#             img = img.transpose(2, 0, 1)
#             img = img / 255 # 归一化
#             if testCount < 0.3 * perClassNum:   # 取30%作测试
#                 testCount +=1
#                 test_x.append(img)
#                 test_y.append(n_class)
#             else:
#                 train_x.append(img)
#                 train_y.append(n_class)
#         n_class += 1
#
#     # # one-hot
#     # lb = LabelBinarizer().fit(np.array(range(n_class)))
#     # train_y = lb.transform(train_y)
#     # test_y = lb.transform(test_y)
#
#     # 转成pytorch数据
#     train_x = torch.tensor(train_x)
#     train_y = torch.tensor(train_y)
#     train_dataset = Data.TensorDataset(train_x, train_y)
#     train_loader = Data.DataLoader(dataset=train_dataset,
#                                    batch_size=batch_size,
#                                    shuffle=True)
#
#     test_x = torch.tensor(test_x)
#     test_y = torch.tensor(test_y)
#     test_dataset = Data.TensorDataset(test_x, test_y)
#     test_loader = Data.DataLoader(dataset=test_dataset,
#                                   batch_size=batch_size,
#                                   shuffle=False)
#     return train_loader, test_loader, n_class
