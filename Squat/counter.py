import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radians *180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle

drawing_spec = mp_drawing.DrawingSpec(thickness=5, circle_radius=4,color = (255,255,255))
drawing_spec_points = mp_drawing.DrawingSpec(thickness=5, circle_radius=4,color = (255,255,255))

display_pos = None
up_pos = None
down_pos = None
counter = 0

vid = cv2.VideoCapture(0)

with mp_pose.Pose( min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1,smooth_landmarks = True ) as pose:

    while vid.isOpened():
        success, image = vid.read()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape
        
        image.flags.writeable = False
        results = pose.process(image)
        eyesVisible = False
        shoulderVisible = False

        try:
            landmarks = results.pose_landmarks.landmark

            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            left_hand = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].y]
            right_hand = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.NOSE.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].visibility = 0

            left_knee_angle = int(calculate_angle(left_hip,left_knee,left_ankle))
            right_knee_angle = int(calculate_angle(right_hip,right_knee,right_ankle))

            if left_knee_angle < 100:
                down_pos = "Down"
                display_pos = "Down"
                        
            if left_knee_angle > 130 and down_pos == "Down":
                up_pos = "Up"
                display_pos = "Up"
            
            if up_pos == "Up" and display_pos == "Up":
                counter += 1
                up_pos = None
                down_pos = None
                
            # print("display_pos : {} down_pos : {} up_pos : {} counter : {}"
            #       .format(display_pos,down_pos,up_pos, counter))


            cv2.putText(image, 'Sayac: ' + str(counter), (image_width-200, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        except:
            pass

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, 
            results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            drawing_spec_points,connection_drawing_spec=drawing_spec)

        final_frame = image

        final_frame = cv2.resize(final_frame,(0,0),fx = 0.5,fy = 0.5)
        cv2.imshow('Frame',final_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    vid.release()
    cv2.destroyAllWindows()