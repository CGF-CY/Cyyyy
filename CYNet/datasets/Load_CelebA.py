import torch
from torch.utils.data import Dataset
from  torch.utils.data import DataLoader
import cv2
import os
import numpy
import logging
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import dlib
class CelebAData(Dataset):
    def __init__(self,root_dir,train=True,target_size=(224,224),transforms=None):
        super(CelebAData, self).__init__()
        self.root_dir=root_dir
        self.target_size=target_size
        self.transforms=transforms
        if train==True:
            image_txt=os.path.join(root_dir,'metas/intra_test/train_label.txt')

        else:
            image_txt=os.path.join(root_dir,'metas/intra_test/test_label.txt')

        self.image_paths=[]
        self.labels=[]
        with open(image_txt,'r') as f:
            for line in f:
                # 移除行末尾的换行符，然后使用空格将行分成两部分
                path, label = line.rstrip('\n').split(' ')
                path = os.path.join(self.root_dir,path)
                # 将路径和标签添加到对应的数组中
                self.image_paths.append(path)
                self.labels.append(int(label))

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(image_path)
        real_h,real_w,c=image.shape
        assert os.path.exists(image_path[:-4]+'_BB.txt'),'path not exists'+' '+image_path
        with open(image_path[:-4]+'_BB.txt','r') as f:
            material =f.readline()
            try:
                x,y,w,h,score=material.split(' ')

            except:
                print('Bounding Box of'+' '+image_path+' '+'is wrong')

            try:
                w=int(float(w))
                h=int(float(h))
                x=int(float(x))
                y=int(float(y))
                w=int(w*(real_w/224))
                h=int(h*(real_h/224))
                x=int(x*(real_w/224))
                y=int(y*(real_h/224))
                y1=0 if y<0 else y
                x1=0 if x<0 else x
                y2=real_h if y1+h>real_h else y+h
                x2=real_w if x1+w>real_w else x+w
                image=image[y1:y2,x1:x2,:]


            except:
                print('Cropping Bounding Box of'+' '+image_path+' '+'goes wrong')

        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.target_size)
        image=transforms.ToTensor()(image)
        if self.transforms is not None:
            image=self.transforms(image)
        return image,label



    def __len__(self):
        return len(self.image_paths)

if __name__ == '__main__':
    train_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test=CelebAData(root_dir="F:\CelebA_Spoof",train=True,transforms=train_transform)
    wirter=SummaryWriter("log")
    loader=DataLoader(test,batch_size=16,shuffle=False,num_workers=2)
    step=0
    for data in loader:
        imgs,labels=data
        wirter.add_images("after",imgs,step)
        step+=1
        if step==50:
            break

    wirter.close()

