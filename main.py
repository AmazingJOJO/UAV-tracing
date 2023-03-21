import cv2
import sys
import json
import os
import numpy as np
from collections import defaultdict
import torch
from FasterRCNN import transform as T
from PIL import Image
def get_next_img(img_path,n):
    imgs = os.listdir(img_path)[n]
    imgs = os.path.join(img_path,imgs)
    return True, cv2.imread(imgs)
def draw_rec(bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    
def img_to_video(path):
    # if os.path.exists("result.mp4"):
    #     print("File already exists")
    #     return
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 
    # frames per second
    cap_fps = 30
    size = (640,512)
    
    # set parameters of video
    video = cv2.VideoWriter('raw_video.mp4', fourcc, cap_fps, size)
    # reading pics
    file_lst = os.listdir(path)
    for filename in file_lst:
        if(filename[-3:] == 'jpg'):
            image = os.path.join(path,filename)
            img = cv2.imread(image)
            video.write(img)
    video.release()

#Using FasterRCNN to reidentify UAV
def reID(frame):
    if_detected = False
    with torch.no_grad():
        img_Image = Image.fromarray(np.uint8(frame))
        img = T.PILToTensor()(img_Image)[0].float()
        prediction = model([img.to(device)])
    if (prediction[0]['boxes'].cpu().shape[0] > 0):
        # print(i)
        xmin = round(prediction[0]['boxes'][0][0].item())
        ymin = round(prediction[0]['boxes'][0][1].item())
        xmax = round(prediction[0]['boxes'][0][2].item())
        ymax = round(prediction[0]['boxes'][0][3].item())
        if_detected = True
    if if_detected == True and (xmax - xmin) / 640 < 0.3 and (ymax - ymin) / 512 < 0.3 and (xmax - xmin) > 0 and (ymax - ymin) > 0 and xmax < 640 and ymax <512:
        # print(xmax - xmin, ymax - ymin)
        cv2.rectangle(frame,  (xmin, ymin), (xmax, ymax), (255,0,0), 2, 1)
        newbbox = (xmin,ymin,xmax-xmin,ymax-ymin)
        return newbbox
    else:
        return False

if __name__ == '__main__' :
    root_path = 'dataset/test/01_2192_0001-1500' #change this to be the dir of files
    print(root_path)
    img_to_video(root_path)

    # Set up tracker.
    tracker = cv2.TrackerCSRT_create()
    # Load FasterRCNN
    model = torch.load(r'saved model/FasterRCNNModel7.pkl')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()
    # Read video
    video = cv2.VideoCapture("raw_video.mp4")

    # Exit if video not opened.
    # if not video.isOpened():
    #     print("Could not open video")
    #     sys.exit()
    
    # Read first frame.
    ok, frame = video.read()
    # if not ok:
    #     print('Cannot read video file')
    #     sys.exit()
    
    # Define an initial bounding box
    imgs = os.listdir(root_path)
    

    json_path = root_path + '/IR_label.json'
    f = open(json_path,'r',encoding='utf-8')
    m = json.load(f) 

    # bbox = tuple(m["gt_rect"][0])
    bbox = tuple(m["res"][0])
    ori_area = bbox[2] * bbox[3]
    ori_bbox = bbox
    record_bbox = bbox
    # bbox = (287, 23, 86, 320)
    
    # Uncomment the line below to select a different bounding box
    # bbox = cv2.selectROI(frame, False)
    
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    flag = True
    freq = 100
    res = {}
    res.setdefault("res",[])

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 
    # frames per second
    cap_fps = 30
    size = (640,512)
    video = cv2.VideoWriter('DEMO.mp4', fourcc, cap_fps, size)
    for i in range(len(imgs) - 1):
        # Read a new frame
        ok, frame = get_next_img(root_path,i)
        
        
        # Start timer
        # timer = cv2.getTickCount()  !!!

        # Update tracker
        ok, bbox = tracker.update(frame)
        
        if ori_area / (640 * 512) > 0.75:
            ok = False
        
        # Calculate Frames per second (FPS)
        # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)  !!!

        # Draw bounding box
        if ok:
            # Tracking success
            # print("1")
            if(flag == False):       #if tracking failure once, detect every frequncy
                if(i % freq == 0):
                    newbbox = reID(frame)
                    if newbbox != False:
                        cv2.putText(frame, "Redetecting", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)  
                        tracker = cv2.TrackerCSRT_create()
                        # print(newbbox)
                        ok = tracker.init(frame, newbbox)
                        res["res"].append(list(newbbox))
                        draw_rec(newbbox)
                        # flag = True
                    else:
                        res["res"].append([])
                else:
                    points = [int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])]
                    res["res"].append(points)
                    draw_rec(points)


            elif i % freq == 0 and i != 0 : #if tracking false target, redetect
                if (abs(ori_bbox[0] - bbox[0]) < 5 or abs(ori_bbox[1] - bbox[1]) < 5):
                    newbbox = reID(frame)
                    if newbbox != False:
                        cv2.putText(frame, "Redetecting", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)  
                        tracker = cv2.TrackerCSRT_create()
                        ok = tracker.init(frame, newbbox)
                        res["res"].append(list(newbbox))
                        draw_rec(newbbox)
                    else:
                        res["res"].append(list(bbox))
                        draw_rec(bbox)
                else :
                    ori_bbox = bbox
                    res["res"].append(list(bbox))
                    draw_rec(bbox)
            else:
                points = [int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])]
                res["res"].append(points)
                draw_rec(points)
        else :
            # Tracking failure
            flag = False
            cv2.putText(frame, "Redetecting", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)  
            
            newbbox = reID(frame)
            if newbbox != False:
                tracker = cv2.TrackerCSRT_create()
                ok = tracker.init(frame, newbbox)
                res["res"].append(list(newbbox))
                draw_rec(newbbox)
            else:
                res["res"].append([])
            # print("2")
            
        # Display result
        cv2.imshow("Tracking", frame) 

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
    # print(len(res['res']))
    # print(res)
    # with open(f'test_result//{folder}.txt', 'w', encoding='utf-8') as f:
    #     f.write(json.dumps(res))
    
    
    # set parameters of video
    
    # reading pics
        video.write(frame)
    video.release()
        
        