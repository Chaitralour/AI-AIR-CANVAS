
import cv2
import mediapipe as mp
import numpy as np
import time

WIDTH, HEIGHT = 1280, 720
TOOLBAR_HEIGHT = 80

COLORS={"RED":(0,0,255),"GREEN":(0,255,0),"BLUE":(255,0,0),
"YELLOW":(0,255,255),"PURPLE":(255,0,255),"BLACK":(0,0,0),
"ERASER":(255,255,255)}

TOOLS=[("RED",20),("GREEN",120),("BLUE",220),("YELLOW",320),
("PURPLE",420),("BLACK",520),("ERASER",620),("CLEAR",760)]

cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,HEIGHT)

mp_hands=mp.solutions.hands
hands=mp_hands.Hands(max_num_hands=2,min_detection_confidence=0.7,min_tracking_confidence=0.7)
drawer=mp.solutions.drawing_utils

canvas=None
draw_color=COLORS["PURPLE"]
prev_x=prev_y=0
prev=time.time()

while True:
    ok,frame=cap.read()
    if not ok:
        break
    frame=cv2.flip(frame,1)
    if canvas is None:
        canvas=np.zeros_like(frame)
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=hands.process(rgb)

    cv2.rectangle(frame,(0,0),(WIDTH,TOOLBAR_HEIGHT),(45,45,45),-1)
    for name,x0 in TOOLS:
        width=100 if name=="CLEAR" else 70
        if name in COLORS:
            cv2.rectangle(frame,(x0,15),(x0+width,65),COLORS[name],-1)
        else:
            cv2.rectangle(frame,(x0,15),(x0+width,65),(90,90,90),-1)
            cv2.putText(frame,name,(x0+8,45),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)

    mode="NONE"
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            drawer.draw_landmarks(frame,hand,mp_hands.HAND_CONNECTIONS)
            h,w,_=frame.shape
            ix=int(hand.landmark[8].x*w)
            iy=int(hand.landmark[8].y*h)
            index_up=hand.landmark[8].y<hand.landmark[6].y
            middle_up=hand.landmark[12].y<hand.landmark[10].y
            cv2.circle(frame,(ix,iy),10,draw_color,-1)

            if index_up and middle_up:
                mode="SELECT"
                prev_x=prev_y=0
                if iy<TOOLBAR_HEIGHT:
                    for t,bx in TOOLS:
                        ww=100 if t=="CLEAR" else 70
                        if bx<=ix<=bx+ww:
                            if t in COLORS:
                                draw_color=COLORS[t]
                            elif t=="CLEAR":
                                canvas=np.zeros_like(frame)
            elif index_up:
                mode="DRAW"
                if prev_x==0 and prev_y==0:
                    prev_x,prev_y=ix,iy
                cv2.line(canvas,(prev_x,prev_y),(ix,iy),draw_color,6,cv2.LINE_AA)
                prev_x,prev_y=ix,iy
            else:
                prev_x=prev_y=0

    out=cv2.add(frame,canvas)
    fps=int(1/max(time.time()-prev,1e-6))
    prev=time.time()
    cv2.putText(out,f"Mode: {mode}",(20,110),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
    cv2.putText(out,f"FPS: {fps}",(1100,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
    cv2.imshow("AirCanvas Pro",out)
    k=cv2.waitKey(1)&0xFF
    if k==ord('q'):
        break
    elif k==ord('s'):
        cv2.imwrite(f"drawing_{int(time.time())}.png",canvas)

cap.release()
cv2.destroyAllWindows()
