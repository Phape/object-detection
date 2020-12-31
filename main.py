import cv2
import time

res_w = 1080
res_h = 720
conf_threshold= 0.4
detection_color= (0,255,0)
configpath = "models/yolov3.cfg"
weightspath = "models/yolov3.weights"
classfile = "models/coco.names"

with open(classfile,"rt") as f:
    classnames = f.read().rstrip("\n").split("\n")

cap = cv2.VideoCapture(0)
cap.set(3, res_w)
cap.set(4, res_h)

net = cv2.dnn_DetectionModel(weightspath,configpath)
net.setInputSize(res_w, res_h)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

while True:
    start = time.time()
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=conf_threshold)
    stop = time.time()
    fps = 1/(stop-start)
    cv2.putText(img, str(round(fps,2)) + ' fps', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, detection_color, 1)
    if len(classIds):
        for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img, box, color=detection_color, thickness=1)
            cv2.putText(img, classnames[classId].upper(),(box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, detection_color, 1)
            cv2.putText(img, str(round(confidence*100))+"%",(box[0]+50,box[1]+100), cv2.FONT_HERSHEY_COMPLEX, 1, detection_color, 1)

    cv2.imshow("Live View", img)
    key = cv2.waitKey(1)
            