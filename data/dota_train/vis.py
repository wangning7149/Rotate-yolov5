import cv2
import os

imgs = os.listdir('./images')
txts = os.listdir('./labels')
for i in imgs:
    im = cv2.imread('./images/' + i)
    with open('./labels/' + i.split('.')[0] + ".txt", 'r') as f:
        lines = f.readlines()
        print(len(lines))
        for l in lines:
            l = l.strip('\n').split(' ')
            if int(l[1]) > im.shape[1]:
                print(i)
            cv2.line(im,(int(l[1]),int(l[2])),(int(l[3]),int(l[4])),(3,0,255),1)
            cv2.line(im, (int(l[3]), int(l[4])), (int(l[5]), int(l[6])), (3, 0, 255), 1)
            cv2.line(im, (int(l[5]), int(l[6])), (int(l[7]), int(l[8])), (3, 0, 255), 1)
            cv2.line(im, (int(l[7]), int(l[8])), (int(l[1]), int(l[2])), (3, 0, 255), 1)
    ima = cv2.resize(im,(0,0),fx=0.5,fy=0.5)
    cv2.imshow("dsa",ima)
    cv2.waitKey()