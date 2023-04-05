import numpy as np
import os
import cv2
import json
import torch
import torch.utils.data as Data
from math import sin, cos, pi
from torch.utils.data import random_split
from tqdm import tqdm
import albumentations as A
from washhand.handpoint_migrate_pytorch import ConvNormActivation,structure

class photoenhance:
    #horizontal_flip = True  # 基本属性,只可在类中更改，不可调用类对象时更改
    def __init__(self,image,keypoints): #hf=True,ra=True,ba=True,sa=True,rna=True):
        self.image = image
        self.keypoints = keypoints
        # self.horizontal_flip = hf
        # self.rotation_augmentation = ra
        # self.brightness_augmentation = ba
        # self.shift_augmentation = sa
        # self.random_noise_augmentation = rna

    # 数据增强
    def Horizontal_flip(self):
        images=self.image
        keypoints=self.keypoints
        transform = A.Compose([A.HorizontalFlip(p=1)], keypoint_params=A.KeypointParams(format='xy'))
        hor_flipped_keypoints = []
        hor_flipped_images = []
        for sample_images, sample_keypoints in zip(images, keypoints):
            #sample_keypoints = sample_keypoints.reshape(16, 2)
            transformed = transform(image=sample_images, keypoints=[tuple(i) for i in sample_keypoints])  #keypoints[[, , ,]]
            hor_flipped_keypoints.append(transformed['keypoints'])  # Subtract only X co-ordinates of keypoints from 96 for horizontal flipping
            hor_flipped_images.append(transformed['image'])
        return hor_flipped_images, hor_flipped_keypoints

    def Vertical_Flip(self):
        images = self.image
        keypoints = self.keypoints
        transform = A.Compose([A.VerticalFlip(p=1)],keypoint_params=A.KeypointParams(format='xy'))
        ver_flipped_keypoints = []
        ver_flipped_images = []
        for sample_images, sample_keypoints in zip(images, keypoints):
            transformed = transform(image=sample_images, keypoints=[tuple(i) for i in sample_keypoints])
            ver_flipped_keypoints.append(transformed['keypoints'])  # Subtract only X co-ordinates of keypoints from 96 for horizontal flipping
            ver_flipped_images.append(transformed['image'])
        return ver_flipped_images,ver_flipped_keypoints


    def blur(self):
        images = self.image
        keypoints = self.keypoints
        transform = A.Compose(
            [A.Blur(blur_limit=15, always_apply=False, p=1)],  # 模糊
            keypoint_params=A.KeypointParams(format='xy')
        )
        blur_keypoints = []
        blur_images = []
        for sample_images, sample_keypoints in zip(images, keypoints):
            transformed = transform(image=sample_images, keypoints=[tuple(i) for i in sample_keypoints])
            blur_keypoints.append(transformed['keypoints'])  # Subtract only X co-ordinates of keypoints from 96 for horizontal flipping
            blur_images.append(transformed['image'])
        return  blur_images,blur_keypoints

    def shift(self):
        images = self.image
        keypoints = self.keypoints
        transform = A.Compose(
            [A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, border_mode=4,
                                value=None, mask_value=None, always_apply=False, p=1)],  # 翻转°
            keypoint_params=A.KeypointParams(format='xy')
        )
        shifted_images = []
        shifted_keypoints = []
        for sample_images, sample_keypoints in zip(images, keypoints):

            transformed = transform(image=sample_images, keypoints=[tuple(i) for i in sample_keypoints])
            shifted_keypoints.append(transformed['keypoints'])  # Subtract only X co-ordinates of keypoints from 96 for horizontal flipping
            shifted_images.append(transformed['image'])
        return  shifted_images,shifted_keypoints

    def gauss_noise(self):
        images = self.image
        keypoints = self.keypoints
        transform = A.Compose(
            [A.GaussianBlur(blur_limit=11, always_apply=False, p=1)],  # 高斯
            keypoint_params=A.KeypointParams(format='xy')
        )
        gauss_images = []
        gauss_keypoints = []
        for sample_images, sample_keypoints in zip(images, keypoints):

            transformed = transform(image=sample_images, keypoints=[tuple(i) for i in sample_keypoints])
            gauss_keypoints.append(transformed['keypoints'])  # Subtract only X co-ordinates of keypoints from 96 for horizontal flipping
            gauss_images.append(transformed['image'])
        return  gauss_images,gauss_keypoints

'''image数据集-pytorch数据集'''
class enhancekeypoint:
    def __init__(self, path,batchsize,h=640,w=640): #必须要有一个self参数，
        self.path = path
        self.batchsize=batchsize
        self.w=w
        self.h=h
    def read_image(self):  # path='C:/Users/Administrator/Desktop/yolov5-master/washhand/handpoint_labels/synth1/'
        '''读取文件夹中的所有图片(关键点)'''
        train_x = []
        train_y = []
        files_point = sorted([f for f in os.listdir(self.path) if f.endswith('.json')])  # os.listdir获取指定文件夹下的所有文件

        # 关键点,图片
        for f in files_point:  # 读取一张图片手坐标(三维坐标点)
            with open(self.path + f, 'r') as fid:
                dat = json.load(fid)  # json文件中读取数据{'hand_points':...,'is_left':...)
            pts = np.array(dat['hand_pts'])
            res=[]
            for i in pts:
                if i[2]==1:
                    res.append([i[0],i[1]])

            path_img = self.path + f[0:-5] + '.jpg'
            img = cv2.imread(path_img, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img = cv2.resize(img, (640, 640))
            transform = A.Compose([A.Resize(width=self.w, height=self.h)],
                                  keypoint_params=A.KeypointParams(format='xy'))
            transformed = transform(image=img, keypoints=[res])
            #img = img.transpose(2, 0, 1)
            #img = img / 255  # 归一化
            train_y.append(transformed['keypoints']) #(?,16,2)
            train_x.append(transformed['image'])
        images = np.array(train_x)
        keypoint_features = np.array(train_y, dtype='float')
        print('成功读取数据')
        #数据增强
        pe=photoenhance(images,keypoint_features)
        #if self.horizontal_flip:
        flipped_train_images, flipped_train_keypoints=pe.Horizontal_flip()
        print("Shape of flipped_train_images:", np.shape(flipped_train_images))
        print("Shape of flipped_train_keypoints:", np.shape(flipped_train_keypoints))
        images = np.concatenate((images, flipped_train_images))
        keypoint_features = np.concatenate((keypoint_features, flipped_train_keypoints))

        #if self.brightness_augmentation:
        Vertical_Flip_train_images, Vertical_Flip_train_keypoints = pe.Vertical_Flip()
        print("Shape of altered_brightness_train_images:", np.shape(Vertical_Flip_train_images))
        print("Shape of altered_brightness_train_keypoints:", np.shape(Vertical_Flip_train_keypoints))
        images = np.concatenate((images, Vertical_Flip_train_images))
        keypoint_features = np.concatenate((keypoint_features, Vertical_Flip_train_keypoints))

        #if self.rotation_augmentation:
        blur_train_images, blur_train_keypoints = pe.blur()
        print("Shape of rotated_train_images:", np.shape(blur_train_images))
        print("Shape of rotated_train_keypoints:", np.shape(blur_train_keypoints))
        images = np.concatenate((images, blur_train_images))
        keypoint_features = np.concatenate((keypoint_features, blur_train_keypoints))

        #if self.shift_augmentation:
        shifted_train_images, shifted_train_keypoints = pe.shift()
        print("Shape of shifted_train_images:", np.shape(shifted_train_images))
        print("Shape of shifted_train_keypoints:", np.shape(shifted_train_keypoints))
        images = np.concatenate((images, shifted_train_images))
        keypoint_features = np.concatenate((keypoint_features, shifted_train_keypoints))

        #if self.random_noise_augmentation:
        noisy_train_images,noisy_train_keypoints = pe.gauss_noise()
        print("Shape of noisy_train_images:", np.shape(noisy_train_images))
        print("Shape of shifted_train_keypoints:", np.shape(noisy_train_keypoints))
        train_images = np.concatenate((images, noisy_train_images))
        train_keypoints = np.concatenate((keypoint_features, noisy_train_keypoints))
        print('成功数据增强')

        # 转成pytorch数据
        # train_x = torch.tensor(train_images)
        # train_y = torch.tensor(train_keypoints)
        # dataset = Data.TensorDataset(train_x, train_y)  # 把训练集和标签继续封装
        # train_data, eval_data = random_split(dataset, [round(0.8 * train_x.shape[0]), round(0.2 * train_x.shape[0])],
        #                                      generator=torch.Generator().manual_seed(42))  # 划分训练和验证
        # traindata = Data.DataLoader(dataset=train_data, batch_size=self.batchsize, shuffle=True, num_workers=0,
        #                             drop_last=True)  # drop_last决定不足一个batch_size的数据是否保留
        # evaldata = Data.DataLoader(dataset=eval_data, batch_size=self.batchsize, shuffle=True, num_workers=0, drop_last=True)

        return train_images, train_keypoints

    def read_image_classify(self):
        '''读取文件夹中的所有图片(分类)'''
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        n_class = 0
        perClassNum = 120   # 每类图片数量
        for child_dir in os.listdir(self.path):  # 类
            child_path = os.path.join(self.path, child_dir)
            print(child_path)
            imgCount = 0
            testCount = 0
            for dir_image in tqdm(os.listdir(child_path)):  # 图片读取
                imgCount += 1
                if imgCount > perClassNum: # 每类用100张
                    break
                img = cv2.imread(child_path + "\\" + dir_image, cv2.IMREAD_COLOR)  # h, w, c
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (640, 640))

                img = img.transpose(2, 0, 1)
                img = img / 255 # 归一化
                if testCount < 0.3 * perClassNum:   # 取30%作测试
                    testCount +=1
                    test_x.append(img)
                    test_y.append(n_class)
                else:
                    train_x.append(img)
                    train_y.append(n_class)
            n_class += 1

        # # one-hot
        # lb = LabelBinarizer().fit(np.array(range(n_class)))
        # train_y = lb.transform(train_y)
        # test_y = lb.transform(test_y)

        # 转成pytorch数据
        train_x = torch.tensor(train_x)
        train_y = torch.tensor(train_y)
        train_dataset = Data.TensorDataset(train_x, train_y)
        train_loader = Data.DataLoader(dataset=train_dataset,
                                       batch_size=self.batchsize,
                                       shuffle=True)

        test_x = torch.tensor(test_x)
        test_y = torch.tensor(test_y)
        test_dataset = Data.TensorDataset(test_x, test_y)
        test_loader = Data.DataLoader(dataset=test_dataset,
                                      batch_size=self.batchsize,
                                      shuffle=False)
        return train_loader, test_loader, n_class


if __name__ == '__main__':
    p = enhancekeypoint('C:/Users/Administrator/Desktop/yolov5-master/washhand/handpoint_labels/synth1/',32)
    image,keypoint=p.read_image()
    model = structure(200, 32,32,1)
    traindata, evaldata = model.data(image, keypoint,p.w,p.h)
    model.train(traindata, evaldata)

