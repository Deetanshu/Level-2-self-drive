# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 10:09:49 2018

@author: deept
"""

#imports
import numpy as np
import argparse
import math
import cv2

#TODO: Distance calculation
def euclidian(a, b):
    return np.linalg.norm(a-b)

#TODO: Get size based on class
def getSize(a, b, c, d, weight):
    return math.sqrt(abs((a-b)*(c-d)))*weight

#TODO: Distance:
def distVeh(a, b):
    return int(a/b)

#TODO: Lane:

def getLane(img, start = 480, change = -1):
    #print("\n-----------Get Lane-----------\nStart Value: ",start,"\nChange: ",change)
    if start < 450 and change == -1:
        return 0,0,0
    if start > 480 and change ==1:
        return 0,0,0
    scanLine = cv2.Canny(img[start:start+2, 300:1000], 100, 200)
    #scanLine = img[start:start+2, 300:1000]
    #print (scanLine, "\n-----------------------------------------------------------")
    #cv2.imshow("Scan Line: ", scanLine)
    i = 349
    j= 351
    while scanLine[1, i] < 1:
        i = i-1
        if i==1:
            return getLane(img, start+change, change)
    while scanLine[1, j] < 1:
        j = j+1
        if j == 698:
            return getLane(img, start+change, change)
    return i,j, start

#Testing:
#img = np.zeros((100, 1200), dtype = int)
#img[51, 303] = 1
#img[50:55, 2] = 1
#img[51, 500] = 1
#print(img)
#getLane(img)

#TODO: Functionality
def detect(net, frame, thresh, fno):
    inWidth = 300
    inHeight = 300
    #WHRatio = inWidth / float(inHeight)
    inScaleFactor = 0.007843
    meanVal = 127.5
    swapRB = True
    classNames = { 0: 'background',
            1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
            7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
            13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
            18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
            24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
            32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
            37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
            41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
            46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
            51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
            56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
            61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
            67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
            75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
            80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
            86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush' }
    blob = cv2.dnn.blobFromImage(frame, inScaleFactor, (inWidth, inHeight), (meanVal, meanVal), swapRB)
    net.setInput(blob)
    detections = net.forward()
    interrupt=False
    cx = 250
    cArr = np.array([cx, 500])
    cars = int(0)
    ppl = int(0)
    cols = frame.shape[1]
    rows = frame.shape[0]
    #cv2.circle(frame, (cx, 450), 4, (0,0,255), -1)
    for i in range(detections.shape[2]):
        if fno==2:
            cArr[0]==150
            cx=150
        confidence = detections[0, 0, i, 2]
        if confidence > thresh:
            class_id = int(detections[0, 0, i, 1])
            #print(class_id)
            if class_id > 19:
                continue
            
            xLeftBottom = int(detections[0, 0, i, 3] * cols)
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop = int(detections[0, 0, i, 5] * cols)
            yRightTop = int(detections[0, 0, i, 6] * rows)
            if class_id == 10 and confidence >0.5:
                if confidence>0.7:
                    interrupt = True
                out = frame[yLeftBottom: yRightTop, xLeftBottom: xRightTop]
                if out.size >0:
                    cv2.imshow("Traffic light watch ", out)
            if class_id >1 and class_id <=9:
                weight = 0.5
            else:
                weight = 1
            size = getSize(xLeftBottom, xRightTop, yLeftBottom, yRightTop, weight)
            cent1 = int((xLeftBottom + xRightTop)/2)
            cent2 = int((yLeftBottom + yRightTop)/2)
            centA = np.array([cent1, yLeftBottom])
            if class_id !=10 and class_id!=1:
                cv2.circle(frame, (cent1, cent2), 4, (0, 255, 0), -1)
            elif class_id==1:
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                              (0, 255, 0))
            if class_id in classNames:
                distEuc = euclidian(cArr, centA)
                #print(distEuc)
                distF = distVeh(distEuc, size)
                if distF < 5:
                    interrupt = True
                label = str(distF)
                if class_id > 9:
                    label = label + " " + classNames[class_id]
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                yLeftBottom = max(yLeftBottom, labelSize[1])
                if class_id != 10 and class_id !=1:
                    cv2.line(frame, (cx, 500), (cent1, cent2), (distF*4,distF*4,255), 2)
                cv2.putText(frame, label, (xLeftBottom, yLeftBottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
                if class_id >= 2 and class_id <=9:
                    cars = cars + 1
                if class_id == 1:
                    ppl = ppl + 1
    #print("Cars: ", cars, "People: ", ppl)
    return frame, cars, ppl, interrupt

#TODO: Split Frame
def splitFrame(frame):
    return frame[:, :500], frame[:, 500:1000], frame[:, 780:1280]

#TODO: Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="test720.mp4", help="Path to video")
    parser.add_argument("--prototxt", default="Model/export_model.pbtxt", help="Path to text network file.")
    parser.add_argument("--weights", default = "Model/frozen_inference_graph.pb", help = "Path to weights.")
    parser.add_argument("--num_classes", default= 90, type=int, help="Number of classes.")
    parser.add_argument("--thr", default=0.2, type=float, help="Default confidence threshold.")
    parser.add_argument("--debug", default = False, type=bool, help="Debug, turned off by default.")
    args = parser.parse_args()
    
    frameno=int(0)
    net = cv2.dnn.readNetFromTensorflow(args.weights, args.prototxt)
    cap = cv2.VideoCapture(args.video)
    cc = np.zeros(3, dtype=int)
    p = np.zeros(3, dtype=int)
    w = np.zeros(3, dtype=int)
    x = [False, False, False]
    original = []
    frames=[]
    st, sp, star, st2, sp2, star2, st_b, sp_b, star_b, st2_b, sp2_b, star2_b=0,0,0,0,0,0,0,0,0,0,0,0
    while True:
        
        back=original
        ret, original = cap.read()
        original = original[:500]

        #cv2.imshow("Canny", can)
        frames=[]
        frame1, frame2, frame3 = splitFrame(original)
        frames.append(frame1)
        frames.append(frame2)
        frames.append(frame3)
        del frame1,frame2,frame3
        d = [False,False,False]

        if frameno%10==0:
            if frameno>2:
                st_b = st
                sp_b = sp
                star_b = star
            del st,sp,star
            st, sp, star=getLane(original)
            st2, sp2, star2=getLane(original, start=400, change= 1)
            if st==0 and sp ==0 and star==0:
                st=st_b
                sp=sp_b
                star=star_b
                #del st_b,sp_b,star_b
            if st2==0 and sp2 ==0 and star2==0:
                st2=st2_b
                sp2=sp2_b
                star2=star2_b
            frames[0], cc[0], p[0], x[0] = detect(net, frames[0],args.thr, 1)
            frames[1], cc[1], p[1], x[1] = detect(net, frames[1], args.thr, 2)
            frames[2], cc[2], p[2], x[2] = detect(net, frames[2], 0.1, 3)

        else:
            for i in range(3):
                w[i] = cc[i] + (p[i]*1.5)
            m = max(w)
            if args.debug:
                d = [True, True, True]
            for i in range(3):
                if x[i]:
                    frames[i],cc[i],p[i],x[i] = detect(net, frames[i], args.thr, (i+1))
                    d[i]=False
                    continue
                if m == w[i] and not x[i]:
                    frames[i],cc[i],p[i],x[i] = detect(net, frames[i], args.thr, (i+1))
                    d[i]=False
                    break
                    #print("Choosing frame ", i, "Weight = ", m)
        
        if d[0] and frameno>1 :
            original[:, :500] = back[:, :500]
        else:
            original[:,:500] = frames[0]
        if d[1] and frameno>1:
            original[:, 500:1000] = back[:, 500:1000]
        else:
            original[:, 500:1000] = frames[1]
        if d[2] and frameno>1:
            original[:, 1000:1280] = back[:, 1000:1280]
        else:
            original[:, 1000:1280] = frames[2][:, 220:]
        #for i in range(3):
         #   outtxt = "Veh: " + str(cc[i]) + " People: " + str(p[i])
          #  cv2.putText(frames[i], outtxt, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) )
           # cv2.imshow("Frame "+str(i), frames[i])
        outtxt = "Veh: " + str((cc[0]+cc[1]))+" People: " + str((p[0]+p[1]))+" Frame:"+ str(frameno)
        cv2.putText(original, outtxt, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255) )
        cv2.line(original, (st+300, star), (st2+300, star2), (255,255,255), 3)
        cv2.line(original, (sp+300, star), (sp2+300, star2), (255,255,255), 3)
        cv2.line(original, (int((st+sp)/2)+300, star), (int((st2+sp2)/2)+300, star2), (0,0,0), 5)
        cv2.imshow("Frame", original)
        if cv2.waitKey(1) >=0:
            break
        frameno= frameno+1
        #print("Frame number : ", frameno)
        
        original = []
    cv2.destroyAllWindows()

                    
                