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

up_pos = None
down_pos = None
pushup_pos = None
display_pos = None

push_up_counter = 0

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

            left_eye = [landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y]
            right_eye = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y]
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,  landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]

            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]

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

            midpoint_shoulder_x = (int(shoulder[0] * image_width )+ int(shoulder_r[0] * image_width))/2
            midpoint_shoulder_y = (int(shoulder[1] * image_height )+ int(shoulder_r[1] * image_height))/2

            midpoint_hip_x = (int(left_hip[0] * image_width )+ int(right_hip[0] * image_width))/2
            midpoint_hip_y = (int(left_hip[1] * image_height)+ int(right_hip[1] * image_height))/2

            based_mid_x = int((midpoint_shoulder_x + midpoint_hip_x)/2)
            based_mid_y = int((midpoint_shoulder_y + midpoint_hip_y)/2)

            neck_point_x = (int(nose[0] * image_width )+ int(midpoint_shoulder_x))/2
            neck_point_y = (int(nose[1] * image_height) + int(midpoint_shoulder_y))/2

            left_arm_angle = int(calculate_angle(shoulder, elbow, wrist))
            right_arm_angle = int(calculate_angle(shoulder_r, elbow_r, wrist_r))
            left_leg_angle = int(calculate_angle(left_hip, left_knee, left_ankle))
            right_leg_angle = int(calculate_angle(right_hip, right_knee, right_ankle))
            left_arm_length = np.linalg.norm(np.array(shoulder) - np.array(elbow))

            mid_point_x = (int(left_hip[0] * image_width )+ int(right_hip[0] * image_width))/2
            mid_point_y = (int(left_hip[1] * image_height)+ int(right_hip[1] * image_height))/2

            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility = 0

            if left_arm_angle > 165:
                up_pos = 'Up'
                display_pos = 'Up'

            if left_arm_angle < 110 and up_pos == 'Up':
                down_pos = 'Down'
                display_pos = 'Down'    

            if left_arm_angle > 165 and down_pos == 'Down':

                pushup_pos = "up"
                display_pos = "up"
                push_up_counter += 1

                up_pos = None
                down_pos = None
                pushup_pos = None 

            cv2.line(image,(int(shoulder[0]* image_width),int(shoulder[1]* image_height)),(int(neck_point_x),int(neck_point_y)),(255,255,255),3)
            cv2.line(image,(int(shoulder_r[0]* image_width),int(shoulder_r[1]* image_height)),(int(neck_point_x),int(neck_point_y)),(255,255,255),3)
            cv2.line(image,(int(shoulder[0]* image_width),int(shoulder[1]* image_height)),(int(elbow[0]* image_width),int(elbow[1]* image_height)),(255,255,255),3)
            cv2.line(image,(int(shoulder_r[0]* image_width),int(shoulder_r[1]* image_height)),(int(elbow_r[0]* image_width),int(elbow_r[1]* image_height)),(255,255,255),3)

            cv2.line(image,(int(neck_point_x),int(neck_point_y)),(int(based_mid_x),int(based_mid_y)),(255,255,255),3,cv2.LINE_4)
            cv2.line(image,(int(based_mid_x),int(based_mid_y)),(int(left_hip[0] * image_width ),(int(left_hip[1] * image_height))),(255,255,255),3,cv2.LINE_8)
            cv2.line(image,(int(based_mid_x),int(based_mid_y)),(int(right_hip[0] * image_width ),(int(right_hip[1] * image_height))),(255,255,255),3,cv2.LINE_8)

            cv2.circle(image,(int(neck_point_x),int(neck_point_y)),4,(255,255,255),5)

            cv2.circle(image,(int(shoulder[0]* image_width),int(shoulder[1]* image_height)),4,(255,255,255),3)
            cv2.circle(image,(int(shoulder_r[0]* image_width),int(shoulder_r[1]* image_height)),4,(255,255,255),3)
            
            cv2.circle(image,(int(based_mid_x),int(based_mid_y)),4,(255,255,255),5)

            cv2.putText(image, 'Sayac: ' + str(push_up_counter), (image_width-200, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

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