#python eyeballcursor_v5.py

from winsound import PlaySound
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import time
import dlib
import cv2
import math
import pyautogui
import win32gui, win32con
from tkinter import *
import pandas as pd

_cameraType = int()
_consecFrame = int()
_arThresh = float()
top = Tk()

print("intial values")
print(_cameraType,_consecFrame,_arThresh)


def Values(cameraType,consecFrame,arThresh):
    global _cameraType,_consecFrame,_arThresh
    _cameraType = cameraType
    _consecFrame = consecFrame
    _arThresh = arThresh 
    top.destroy()
    print(cameraType,consecFrame,arThresh)

def MainWndow():
    #creating the application main window.
    #top = Tk()
    top.geometry("450x250")
    top.title("Eye")

    labelframe = LabelFrame(top, text="Prerequisite")
    labelframe.pack(fill="both", expand="yes")
    Label(top, text = "Select Camera :").place(x = 30,y = 50)
    cameraType = IntVar()

    Radiobutton(top, text="Web camera", variable=cameraType, value=0).place(x = 140,y = 50)
    Radiobutton(top, text="USB camera", variable=cameraType, value=1).place(x = 250,y = 50)

    Label(top, text = "Select EYE_AR_CONSEC_FRAMES :").place(x = 30,y = 80)
    consecFrame=Spinbox(top, from_= 1, to = 10,width=5,)
    consecFrame.place(x = 250,y = 80)


    Label(top, text = "Select EYE_AR_THRESH :").place(x = 30,y = 110)
    arThresh=Spinbox(top, from_= 1.2, to = 2.0,format="%0.2f",increment=0.01,width=5)
    arThresh.place(x = 250,y = 110)


    Button(top,text = "Start",command = lambda : Values(cameraType.get(),float(consecFrame.get()),float(arThresh.get()))).place(x = 200,y = 150)

    top.mainloop()

MainWndow()  #calling main window


    
print("after button press values")
print(type(_cameraType),type(_consecFrame),type(_arThresh))

pyautogui.FAILSAFE = False
Minimize = win32gui.GetForegroundWindow()
win32gui.ShowWindow(Minimize, win32con.SW_MINIMIZE)

screen_rez = pyautogui.size()
rez_x = screen_rez[0]
rez_y = screen_rez[1]
calib_rez_x = int(0.75*rez_x)
calib_rez_y = int(0.75*rez_y)
frame_x = int(rez_x/2)
frame_y = int(rez_y/2)

#print("Two camera options")
#print("")
#print("Please press 0 for Web camera")
#print("Please press 1 for USB camera")
#print("") 
#print("")

#camera_option = input("Enter camera number:  ")
#while 
#camera_option = 2
#camera_option = int(camera_option)
def sound_alarm(path):
    # play an alarm sound
    PlaySound.playsound(path)
#for csv file
xcord_cs = []
ycord_cs = []
xmouse_cs = []
ymouse_cs = []
def eye_aspect_ratio(eye):

    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    EYE = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    cnter=(eye[0]+eye[3])/2;
    return EYE,cnter

EYE_AR_CONSEC_FRAMES = _consecFrame  #3 to 10
print(EYE_AR_CONSEC_FRAMES)
COUNTER = 0
ALARM_ON = False

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor

cam = cv2.VideoCapture(_cameraType)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
time.sleep(1.0)


muldiffcentX=0
muldiffcentY=0
CentInitX=rez_x/2
CentInitY=rez_y/2
FrameCentX=frame_x/2
FrameCentY=frame_y/2
lp=0



def CalibrationHelper(displaypointX, displaypointY, displayString):
    
    blank_image = 255 * np.ones(shape=[calib_rez_y, calib_rez_x, 3], dtype=np.uint8)
    cv2.circle(blank_image,(displaypointX,displaypointY),15,(0,255,0),-1)
    sttime = time.time()
    cv2.putText(blank_image, displayString, (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    while time.time()-sttime < 2:
        cv2.imshow("Calibration Frame", blank_image)
        cv2.moveWindow("Calibration Frame", int(rez_x/2) - int(calib_rez_x/2), int(rez_y/2) - int(calib_rez_y/2) - 35)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    Xarr = []
    Yarr = []
    EAR_arr = []
    blank_image = 255 * np.ones(shape=[calib_rez_y, calib_rez_x, 3], dtype=np.uint8)
    cv2.circle(blank_image,(displaypointX,displaypointY),15,(0,255,0),-1)
    sttime = time.time()
    while time.time()-sttime < 4:  #3 to 5 changed 4
                
        __,frame = cam.read()
        
        frame = cv2.resize(frame,(frame_x, frame_y), interpolation = cv2.INTER_CUBIC)
        #frame = cv2.flip(frame,0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR,cnter = eye_aspect_ratio(leftEye)
            rightEAR,cnterr = eye_aspect_ratio(rightEye)
            xl = int(math.ceil(cnter[0]));
            yl = int(math.ceil(cnter[1]));
            xr = int(math.ceil(cnterr[0]));
            yr = int(math.ceil(cnterr[1]));
            x = int((xl+xr)/2)
            y = int((yl+yr)/2)
            cv2.circle(frame,(x,y),5,255,-1);
            EAR_arr.append((leftEAR+rightEAR)/2)
            Xarr.append(x)
            Yarr.append(y)
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        cv2.imshow("Frame", frame)
        cv2.imshow("Calibration Frame", blank_image)
        cv2.moveWindow("Calibration Frame", int(rez_x/2) - int(calib_rez_x/2), int(rez_y/2) - int(calib_rez_y/2) - 35)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break    
    cv2.destroyWindow("Calibration Frame")
    return Xarr,Yarr,EAR_arr

def Calibration():    
    x0_arr, y0_arr, EAR_arr = CalibrationHelper(int(calib_rez_x/2),int(calib_rez_y/2),"Please Look at the appearing dot and keep \n both your eyes blinked for 3 seconds")
    x0 = int(sum(x0_arr)/len(x0_arr))
    y0 = int(sum(y0_arr)/len(y0_arr))
    #print(x0_arr)
    #print(y0_arr)
    EYE_AR_THRESH = sum(EAR_arr)/len(EAR_arr)
    x1_arr, y1_arr, __ = CalibrationHelper(int(calib_rez_x/2),0,"Please Look at the appearing dot for 3 seconds")
    x1 = int(sum(x1_arr)/len(x1_arr))
    y1 = int(sum(y1_arr)/len(y1_arr))
    #print(x1_arr, y1_arr) #by me 13/09/21
    print(x1, y1)
    mul_y1 = (calib_rez_y/2)/abs(y0-y1)
    x2_arr, y2_arr, __ = CalibrationHelper(int(calib_rez_x/2),calib_rez_y,"Please Look at the appearing dot for 3 seconds")
    x2 = int(sum(x2_arr)/len(x2_arr))
    y2 = int(sum(y2_arr)/len(y2_arr))
    #print(x2_arr, y2_arr)  #by me 13/09/21
    mul_y2 = (calib_rez_y/2)/abs(y0-y2)
    print(x2, y2)
    x3_arr, y3_arr, __ = CalibrationHelper(0,int(calib_rez_y/2),"Please Look at the appearing dot for 3 seconds")
    x3 = int(sum(x3_arr)/len(x3_arr))
    y3 = int(sum(y3_arr)/len(y3_arr))
    print(x3, y3) #before upr sab me tha..edit kiya mene
    #print(x3_arr,  y3_arr) #by me 13/09/21
    mul_x3 = (calib_rez_x/2)/abs(x0-x3)
    x4_arr, y4_arr, __ = CalibrationHelper(calib_rez_x,int(calib_rez_y/2),"Please Look at the appearing dot for 3 seconds")
    x4 = int(sum(x4_arr)/len(x4_arr))
    y4 = int(sum(y4_arr)/len(y4_arr))
    print(x4, y4) #before upr sab me tha..edit kiya mene
    #print(x4_arr, y4_arr) #by me 13/09/21
    mul_x4 = (calib_rez_x/2)/abs(x0-x4)
    return x0, y0, mul_y1, mul_y2, mul_x3, mul_x4, EYE_AR_THRESH

x0, y0, mul_y1, mul_y2, mul_x3, mul_x4, EYE_AR_THRESH = Calibration()
mul_y1 = 0.75 * mul_y1
mul_y2 = 0.75 * mul_y2
mul_x3 = 0.75 * mul_x3
mul_x4 = 0.75 * mul_x4
if EYE_AR_THRESH > 0.10:
    print(type(EYE_AR_THRESH),type(_arThresh))
    EYE_AR_THRESH = EYE_AR_THRESH/_arThresh
else:
    EYE_AR_THRESH = EYE_AR_THRESH

#new updation from here...

#EYE_AR_THRESH = float(EYE_AR_THRESH)
   
 
print(x0, y0, mul_y1, mul_y2, mul_x3, mul_x4, EYE_AR_THRESH) 
         
#pyautogui.hotkey('win', 'ctrl', 'o')

#webbrowser.open_new('https://eye-tracker-d8638.web.app/')
# webbrowser.open_new('https://shobhit5923.github.io/trackk.github.io/index.html')
#csv file making.......

while True:
    
    __,frame = cam.read()    
    frame = cv2.resize(frame,(frame_x, frame_y), interpolation = cv2.INTER_CUBIC)
    #frame = cv2.flip(frame,0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    
    for rect in rects:

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR,cnter = eye_aspect_ratio(leftEye)
        rightEAR,cnterr = eye_aspect_ratio(rightEye)
        xl = int(math.ceil(cnter[0]));
        yl = int(math.ceil(cnter[1]));
        xr = int(math.ceil(cnterr[0]));
        yr = int(math.ceil(cnterr[1]));
        x = int((xl+xr)/2)
        y = int((yl+yr)/2)
        xcord_cs.append(x)
        ycord_cs.append(y)
        #print(xcord_cs, ycord_cs)
        cv2.circle(frame,(int(FrameCentX),int(FrameCentY)),5,(0,255,0),-1)
        cv2.circle(frame,(x,y),5,255,-1)
        if lp>0:
            diffcentX=x0-x;
            diffcentY=y0-y;
            #10,15   12,18    P20,25    15,20
            if(y<y0):
                muldiffcentY=diffcentY*mul_y1
            elif(y>=y0):
                muldiffcentY=diffcentY*mul_y2
            if(x<x0):
                muldiffcentX=diffcentX*mul_x3
            elif(x>=x0):
                muldiffcentX=diffcentX*mul_x4
        moveCordX = CentInitX+muldiffcentX
        moveCordY = CentInitY-muldiffcentY
        #xmouse_cs.append(moveCordX) #by me 13/09/2021 commemt
        #ymouse_cs.append(moveCordY) #by me 13/09/2021 comment
        #print(moveCordX, moveCordY)
        if moveCordX>rez_x:
            moveCordX = rez_x
        elif moveCordX<0:
            moveCordX = 0
        if moveCordY>rez_y:
            moveCordY = rez_y
        elif moveCordY<0:
            moveCordY = 0
        if COUNTER == 0 and leftEAR > EYE_AR_THRESH and rightEAR > EYE_AR_THRESH:
            pyautogui.moveTo(moveCordX, moveCordY)

        
        lp=1

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if (leftEAR < EYE_AR_THRESH) or (leftEAR < EYE_AR_THRESH and rightEAR < EYE_AR_THRESH):
            pyautogui.doubleClick() 
            COUNTER += 1

            # if the eyes were closed for a sufficient number of
            # then sound the alarm
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                pyautogui.click(button='left')
                COUNTER = 0
                # if the alarm is not on, turn it on
                if not ALARM_ON:
                    ALARM_ON = True

                # draw an alarm on the frame
                cv2.putText(frame, "LEFT CLICK PRESSED!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        elif rightEAR < EYE_AR_THRESH:
            COUNTER += 1
            
            # if the eyes were closed for a sufficient number of
            # then sound the alarm
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                pyautogui.click(button='right')
                COUNTER = 0
                # if the alarm is not on, turn it on
                if not ALARM_ON:
                    ALARM_ON = True


                # draw an alarm on the frame
                cv2.putText(frame, "RIGHT CLICK PRESSED!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # otherwise, the eye aspect ratio is not below the blink
        # threshold, so reset the counter and alarm
        else:
            COUNTER = 0
            ALARM_ON = False

        cv2.putText(frame, "Left Eye: {:.2f}, Right Eye: {:.2f}".format(leftEAR, rightEAR), (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()



