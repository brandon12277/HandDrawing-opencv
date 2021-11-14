import cv2 as cv
import mediapipe as mp
import numpy as np

class Hand_Drawing():

    def __init__(self, mode=False, Hands_num=1, Track_con=2, detection_con=0.5):
        self.Hands_num = Hands_num
        self.mode = mode
        self.Track_con = Track_con
        self.detection_con = detection_con
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def FindHands(self, frame):
        cvtrgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.handstracked = self.hands.process(cvtrgb)
        if self.handstracked.multi_hand_landmarks:
            for hand_marks in self.handstracked.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(frame, hand_marks, self.mpHands.HAND_CONNECTIONS)
        return frame

    def DisplayPoints(self, frame):
        lmst = []
        if self.handstracked.multi_hand_landmarks:
            hand = self.handstracked.multi_hand_landmarks
            for hand_marks in hand:
                for id, lm in enumerate(hand_marks.landmark):
                    h, w, c = frame.shape
                    cx1 = (int)(lm.x * w)
                    cy1 = (int)(lm.y * h)
                    lmst.append([id, cx1, cy1])
            return lmst


def main():
    cam = cv.VideoCapture(0)
    drawer = Hand_Drawing()
    drawing_points=[]
    color=(0,0,0)
    drawing_screen = np.full((480,640,3), 255,dtype="uint8")
    black=np.full((480,640,3), 0,dtype="uint8")
    #controls for drawing
    draw_x=0
    draw_y=0
    frame_count = 0
    while True:
        success, frame = cam.read()
        frame = drawer.FindHands(frame)
        cv.rectangle(frame, (0, 0), (150, 50), (255, 255, 255),-1)
        cv.line(frame,(150,0),(150,50),(0,0,0),2)
        cv.putText(frame,"Green",(75,25),cv.FONT_HERSHEY_PLAIN, 1,(0,0,0))
        cv.rectangle(frame, (150, 0), (300, 50), (255, 255, 255),-1)
        cv.line(frame, (300, 0), (300, 50), (0, 0, 0), 2)
        cv.putText(frame, "Blue", (200, 25), cv.FONT_HERSHEY_PLAIN, 1,(0,0,0))
        cv.rectangle(frame, (300, 0), (450, 50), (255, 255, 255),-1)
        cv.line(frame, (450, 0), (450, 50), (0, 0, 0), 2)
        cv.putText(frame, "Red", (325, 25), cv.FONT_HERSHEY_PLAIN, 1,(0,0,0))
        cv.rectangle(frame, (450, 0), (640, 50), (255, 255, 255),-1)
        cv.putText(frame, "Clear Screen", (500, 25), cv.FONT_HERSHEY_PLAIN, 1,(0,0,0))
        lmst=drawer.DisplayPoints(frame)
        if(lmst):
            frame_count += 1
            for id,x,y in lmst:
                if(id==8):
                    cv.circle(frame,(x,y),10,(255,0,0),-1)
                    if (x >= 0 and x <= 150 and y >= 0 and y <= 50):
                        color = (0, 255, 0)
                    if (x >= 150 and x <= 300 and y >= 0 and y <= 50):
                        color = (255, 0, 0)
                    if (x >= 300 and x <= 350 and y >= 0 and y <= 50):
                        color = (0, 0, 255)
                    if (x >= 450 and x <= 640 and y >= 0 and y <= 50):
                        drawing_points = []
                    if(x>=0 and x<=640 and y>=0 and y<=50):
                        break
                    else:
                        drawing_points.append([x,y,color])

        for i in range(1, len(drawing_points)):
            if (drawing_points[i - 1] is None or drawing_points[i] is None):
                continue;
            cv.line(frame, (drawing_points[i - 1][0],drawing_points[i - 1][1]), (drawing_points[i][0],drawing_points[i][1])
                    , drawing_points[i][2], 2)
        cv.imshow("Video", frame)
        cv.imshow("drawing_screen",drawing_screen)
        if cv.waitKey(20) & 0xFF == ord('d'):
            break


if __name__ == "__main__":
    main()