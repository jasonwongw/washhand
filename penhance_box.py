import numpy as np
import os
import cv2
import pandas as pd
import albumentations as A
from handpoint_migrate_pytorch import ConvNormActivation,structure

def savecsv(data,name):
    file_X = name
    datax = pd.DataFrame(data)
    datax.to_csv(file_X, index=False)

class photoenhance:
    #horizontal_flip = True  # 基本属性,只可在类中更改，不可调用类对象时更改
    def __init__(self,image,boxes,category_ids):
        self.image = image
        self.boxes = boxes
        self.category_ids=category_ids

    # 数据增强
    def Horizontal_flip(self):
        images=self.image
        boxes=self.boxes
        category_ids=self.category_ids
        transform = A.Compose([A.HorizontalFlip(p=1)], bbox_params= A.BboxParams(format='yolo', min_area=500, min_visibility=0.1,label_fields=['category_ids']))
        hor_flipped_box = []
        hor_flipped_images = []
        hor_filpped_id=[]
        for sample_image, sample_box,sample_id in zip(images, boxes,category_ids):
            transformed = transform(image=sample_image, bboxes=[sample_box],category_ids=sample_id)  #bboxes:[[, , ,]]
            hor_flipped_box.append(np.array(transformed['bboxes']).flatten())  # Subtract only X co-ordinates of keypoints from 96 for horizontal flipping
            hor_flipped_images.append(transformed['image'])
            hor_filpped_id.append(transformed['category_ids'])
        return hor_flipped_images, hor_flipped_box,hor_filpped_id

    def Vertical_Flip(self):
        images = self.image
        boxes = self.boxes
        category_ids = self.category_ids
        transform = A.Compose([A.VerticalFlip(p=1)],
                              bbox_params=A.BboxParams(format='yolo', min_area=500, min_visibility=0.1,
                                                       label_fields=['category_ids']))
        ver_flipped_box = []
        ver_flipped_images = []
        ver_filpped_id = []
        for sample_image, sample_box, sample_id in zip(images, boxes, category_ids):
            transformed = transform(image=sample_image, bboxes=[sample_box], category_ids=sample_id)
            ver_flipped_box.append(np.array(transformed[
                                                'bboxes']).flatten())  # Subtract only X co-ordinates of keypoints from 96 for horizontal flipping
            ver_flipped_images.append(transformed['image'])
            ver_filpped_id.append(transformed['category_ids'])
        return ver_flipped_images, ver_flipped_box, ver_filpped_id


    def blur(self):
        images = self.image
        boxes = self.boxes
        category_ids = self.category_ids
        transform = A.Compose([A.Blur(blur_limit=15, always_apply=False, p=1)],
                              bbox_params=A.BboxParams(format='yolo', min_area=500, min_visibility=0.1,
                                                       label_fields=['category_ids']))
        blur_box = []
        blur_images = []
        blur_id = []
        for sample_image, sample_box, sample_id in zip(images, boxes, category_ids):
            transformed = transform(image=sample_image, bboxes=[sample_box], category_ids=sample_id)
            blur_box.append(np.array(transformed['bboxes']).flatten())  # Subtract only X co-ordinates of keypoints from 96 for horizontal flipping
            blur_images.append(transformed['image'])
            blur_id.append(transformed['category_ids'])
        return blur_images, blur_box, blur_id


    def shift(self):
        images = self.image
        boxes = self.boxes
        category_ids = self.category_ids
        transform = A.Compose([A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, border_mode=4,
                                value=None, mask_value=None, always_apply=False, p=1)],
                              bbox_params=A.BboxParams(format='yolo', min_area=500, min_visibility=0.1,
                                                       label_fields=['category_ids']))
        shift_box = []
        shift_images = []
        shift_id = []
        for sample_image, sample_box, sample_id in zip(images, boxes, category_ids):
            transformed = transform(image=sample_image, bboxes=[sample_box], category_ids=sample_id)
            shift_box.append(np.array(transformed['bboxes']).flatten())  # Subtract only X co-ordinates of keypoints from 96 for horizontal flipping
            shift_images.append(transformed['image'])
            shift_id.append(transformed['category_ids'])
        return shift_images, shift_box, shift_id

    def gauss_noise(self):
        images = self.image
        boxes = self.boxes
        category_ids = self.category_ids
        transform = A.Compose([A.GaussianBlur(blur_limit=11, always_apply=False, p=1)],
                              bbox_params=A.BboxParams(format='yolo', min_area=500, min_visibility=0.1,
                                                       label_fields=['category_ids']))
        gauss_box = []
        gauss_images = []
        gauss_id = []
        for sample_image, sample_box, sample_id in zip(images, boxes, category_ids):
            transformed = transform(image=sample_image, bboxes=[sample_box], category_ids=sample_id)
            gauss_box.append(np.array(transformed['bboxes']).flatten())  # Subtract only X co-ordinates of keypoints from 96 for horizontal flipping
            gauss_images.append(transformed['image'])
            gauss_id.append(transformed['category_ids'])
        return gauss_images, gauss_box, gauss_id


class enhancebox:
    def __init__(self,path,w=320,h=320):  #w,h：rezise后的宽高
        self.path=path
        self.w=w
        self.h=h
    def read_image(self):  # path='C:\Users\Administrator\Desktop\handdata\HandWashDataset\train_label'
        '''读取文件夹中的所有图片(关键点)'''
        path=self.path
        train_x = []
        train_y = []
        c=[]
        files_point = sorted([f for f in os.listdir(path) if f.endswith('.txt')])  # os.listdir获取指定文件夹下的所有文件
        # 关键点,图片
        for f in files_point:  # 读取一张图片手坐标(三维坐标点)
            with open(path + f, 'r') as fid:
                dat = np.loadtxt(fid)  # json文件中读取数据{'hand_points':...,'is_left':...)
                c.append(dat[:1])
                path_img = 'train/' +f[:-4] + '.jpg'
                img = cv2.imread(path_img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #修改图片尺寸，作为原始数据并修改标记框
                transform = A.Compose([A.Resize(width=self.w, height=self.h)],bbox_params=A.BboxParams(format='yolo', min_area=500, min_visibility=0.1,label_fields=['category_ids']))
                transformed = transform(image=img, bboxes=[list(dat[1:])], category_ids=dat[:1])
                # img = img.transpose(2, 0, 1)
                #img = img / 255  # 归一化
                train_x.append(transformed['image'])
                train_y.append(transformed['bboxes'])
        images = np.array(train_x) #如果不能将长度不一的列表变为数组，就执行下面一句。并将下面所有数据增强的列表，改为numpy形式(dtype=object)后再concatenate
        #images = np.array(train_x,dtype=object)
        box_features = np.array(train_y, dtype='float')  ## (?,1,4)
        box_features = box_features.reshape(-1, 4)
        category_id=np.array(c)
        #category_id_to_name = {0: '内',1:'外',2:'夹',3:'弓',4:'大',5:'立',6:'腕' }
        pe=photoenhance(images,box_features,category_id)
        flipped_train_images, flipped_train_boxes,flipped_train_id = pe.Horizontal_flip()

        print("Shape of flipped_train_images:", np.shape(flipped_train_images))
        print("Shape of flipped_train_boxes:", np.shape(flipped_train_boxes)) #(572, 4)
        print("Shape of flipped_train_id:", np.shape(flipped_train_id))
        #flipped_train_images=np.array(flipped_train_images,dtype=object)
        images = np.concatenate((images, flipped_train_images), axis = 0)
        box_features = np.concatenate((box_features, flipped_train_boxes))

        category_id = np.concatenate((category_id, flipped_train_id))
        print('----{}-----'.format(len(images)))

        ver_train_images, ver_train_boxes, ver_train_id = pe.Vertical_Flip()
        print("Shape of ver_train_images:", np.shape(ver_train_images))
        print("Shape of ver_train_boxes:", np.shape(ver_train_boxes))
        print("Shape of ver_train_id:", np.shape(ver_train_id))

        ver_train_images = np.array(ver_train_images, dtype=object)
        images = np.concatenate((images, ver_train_images))
        box_features = np.concatenate((box_features, ver_train_boxes))
        category_id = np.concatenate((category_id, ver_train_id))
        print('----{}-----'.format(len(images)))

        blur_train_images, blur_train_boxes, blur_train_id = pe.blur()
        print("Shape of blur_train_images:", np.shape(blur_train_images))
        print("Shape of blur_train_boxes:", np.shape(blur_train_boxes))
        print("Shape of blur_train_id:", np.shape(blur_train_id))
        #blur_train_images = np.array(blur_train_images, dtype=object)
        images = np.concatenate((images, blur_train_images))
        box_features = np.concatenate((box_features, blur_train_boxes))
        category_id = np.concatenate((category_id, blur_train_id))
        print('----{}-----'.format(len(images)))

        shitf_train_images, shitf_train_boxes, shitf_train_id = pe.shift()
        print("Shape of shitf_train_images:", np.shape(shitf_train_images))
        print("Shape of shitf_train_boxes:", np.shape(shitf_train_boxes))
        print("Shape of shitf_train_id:", np.shape(shitf_train_id))
        #shitf_train_images = np.array(shitf_train_images, dtype=object)
        images = np.concatenate((images, shitf_train_images))
        box_features = np.concatenate((box_features, shitf_train_boxes))
        category_id = np.concatenate((category_id, shitf_train_id))
        print('----{}-----'.format(len(images)))

        gauss_train_images, gauss_train_boxes, gauss_train_id = pe.gauss_noise()
        print("Shape of gauss_train_images:", np.shape(gauss_train_images))
        print("Shape of gauss_train_boxes:", np.shape(gauss_train_boxes))
        print("Shape of gauss_train_id:", np.shape(gauss_train_id))
        #gauss_train_images = np.array(gauss_train_images, dtype=object)
        images = np.concatenate((images, gauss_train_images))
        box_features = np.concatenate((box_features, gauss_train_boxes))
        category_id = np.concatenate((category_id, gauss_train_id))
        print('----{}-----'.format(len(images)))
        print('数据增强和标记框修改成功')
        return images,box_features,category_id


if __name__ == '__main__':
    #p=enhancebox('train_label/')
    # pe,b,id=p.read_image()

    # #img=pe.reshape()  #此时pe是三维的，需要降维才能保存
    # #txt = np.concatenate((id, b), axis=1)
    # # savecsv(pe,'pe.csv')
    # # savecsv(txt, 'txt.csv')

    model=structure(20,32,4,1)
    # traindata,evaldata=model.data(pe,b,p.w,p.h) #改变rezise的w,h
    # model.train(traindata,evaldata)

    #预测并显示
    model.predict_show('/tmp/pycharm_project_201/best_handbox.pth','/tmp/pycharm_project_201/hand.jpg')