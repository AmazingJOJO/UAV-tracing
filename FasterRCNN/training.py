import utils
import transform as T
from engine import train_one_epoch, evaluate
import torch
from dataset import mydataset,testdataset
import torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

# utils、transforms、engine就是刚才下载下来的utils.py、transforms.py、engine.py
 
def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.PILToTensor())
    # if train:
    #     # during training, randomly flip the training images
    #     # and ground-truth for data augmentation
    #     # 50%的概率水平翻转
    #     transforms.append(T.RandomHorizontalFlip(0.5))
 
    return T.Compose(transforms)
def train(num_epochs,num_classes,dataset,dataset_test,device):
    

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, # num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=2, shuffle=False, # num_workers=4,
        collate_fn=utils.collate_fn)
    # get the model using our helper function
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=num_classes, pretrained_backbone=True)  # 或get_object_detection_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    # SGD
    optimizer = torch.optim.SGD(params, lr=0.0003,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler
    # cos学习率
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

    # let's train it for   epochs
    

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        # engine.py的train_one_epoch函数将images和targets都.to(device)了
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset    
        evaluate(model, data_loader_test, device=device)    
        
        print('')
        print('==================================================')
        print('')

    print("That's it!")

    torch.save(model, r'../saved_model/FasterRCNNModel200.pkl')

def showbbox(model, img,device,num_bbox):
    # 输入的img是0-1范围的tensor        
    model.eval()
    with torch.no_grad():
        '''
        prediction形如:
        [{'boxes': tensor([[1492.6672,  238.4670, 1765.5385,  315.0320],
        [ 887.1390,  256.8106, 1154.6687,  330.2953]], device='cuda:0'), 
        'labels': tensor([1, 1], device='cuda:0'), 
        'scores': tensor([1.0000, 1.0000], device='cuda:0')}]
        '''
        print(img.type)
        img = img.float()
        prediction = model([img.to(device)])
        
    # print(prediction)
        
    img = img.permute(1,2,0)  # C,H,W → H,W,C，用来画图
    img = (img).byte().data.cpu()  # * 255，float转0-255
    img = np.array(img)  # tensor → ndarray
    num = 0
    for i in range(prediction[0]['boxes'].cpu().shape[0]):
        xmin = round(prediction[0]['boxes'][i][0].item())
        ymin = round(prediction[0]['boxes'][i][1].item())
        xmax = round(prediction[0]['boxes'][i][2].item())
        ymax = round(prediction[0]['boxes'][i][3].item())
        
        label = prediction[0]['labels'][i].item()
        
        if label == 1:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), thickness=1)
            cv2.putText(img, 'UAV', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                            thickness=1)
        
        num += 1
        if (num == num_bbox):
            break
    
    plt.figure(figsize=(20,15))
    plt.imshow(img)
    plt.show()

def test():
    model = torch.load(r'../FasterRCNN/FasterRCNNModel2.pkl')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    newtest = testdataset('../dataset/test',get_transform(train=False))
    
    model.to(device)
    for i in range(0,1):
        num = random.randint(0,len(newtest) - 1)
        img = newtest[num]

        showbbox(model,img,device,1)

if __name__ == '__main__':
    root = r'../dataset/train'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # use our dataset and defined transformations
    dataset = mydataset(root, get_transform(train=True))
    # dataset_test = mydataset(root, get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    train_num = int(0.8 * len(dataset))
    
    dataset_train = torch.utils.data.Subset(dataset, indices[:train_num])
    dataset_test = torch.utils.data.Subset(dataset, indices[train_num:])
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # train(200,2,dataset,dataset_test,device=device)
    test()
    