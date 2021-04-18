import queue
from scipy.spatial import distance as dist
from imutils.video import VideoStream, FPS
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import pandas as pd
from datetime import datetime
from pyimagesearch.centroidtracker import CentroidTracker


def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5]) 
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def mouth_aspect_ratio(mouth):
	A = dist.euclidean(mouth[2], mouth[10])
	B = dist.euclidean(mouth[4], mouth[8])
	C = dist.euclidean(mouth[3], mouth[9])
	D = dist.euclidean(mouth[0], mouth[6])
	mar = (A + B + C) / (3.0 * D)
	return mar

def generate_csv(list_time):
	to_csv_file = pd.DataFrame(list_time)
	now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
	to_csv_file.to_csv('C:\\Users\\ahsan\\Documents\\res\\{}.csv'.format(now))

def millis_convert(millis):
	minutes = int((millis/(1000*60))%60)
	seconds = int((millis/1000)%60)
	return '{}:{}'.format(minutes, seconds)

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",required=True, help="path to input video file")
args = vars(ap.parse_args())

ct = CentroidTracker()
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 15
MOUTH_AR_THRESH = 0.83

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = (49, 68)

vs = cv2.VideoCapture(args["video"])
fps = FPS().start()

face_list = []
exec_time_30_frame = queue.Queue(maxsize=15)
exec_time_list = []
vs_time = 0

while True:
	vs_time = int(vs.get(cv2.CAP_PROP_POS_MSEC))

	e1 = cv2.getTickCount()

	(grabbed, frame) = vs.read()
	
	cv2.putText(frame, millis_convert(vs_time), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

	if not grabbed:
		generate_csv(exec_time_list)
		break

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)
	unordered_rects = []

	for rect in rects:
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		startX = rect.left()
		startY = rect.top()
		endX = rect.right()
		endY = rect.bottom()
		cX = int((startX + endX) / 2.0)
		cY = int((startY + endY) / 2.0)
		unordered_rects.append([startX, startY, endX, endY, [cX, cY]])

	objects_ordered = ct.update(unordered_rects)

	for (objectID, centroid) in objects_ordered.items():
		try:
			face_list[objectID]
		except IndexError:
			face_list.append({
				'consec_frame': 0,
			})

		for u in unordered_rects:
			if centroid[0] == u[4][0] and centroid[1] == u[4][1]:
				rect = dlib.rectangle(u[0], u[1], u[2], u[3])

				shape = predictor(gray, rect)
				shape = face_utils.shape_to_np(shape)

				leftEye = shape[lStart:lEnd]
				rightEye = shape[rStart:rEnd]
				leftEAR = eye_aspect_ratio(leftEye)
				rightEAR = eye_aspect_ratio(rightEye)
				ear = (leftEAR + rightEAR) / 2.0

				leftEyeHull = cv2.convexHull(leftEye)
				rightEyeHull = cv2.convexHull(rightEye)
				cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
				cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

				mouth = shape[mStart:mEnd]
				mar = mouth_aspect_ratio(mouth)

				mouthHull = cv2.convexHull(mouth)	
				cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

				(x, y, w, h) = face_utils.rect_to_bb(rect)
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
				cv2.putText(frame, '#{} E:{:.2f} M:{:.2f}'.format(objectID, ear, mar), (x - 10, y - 4), 
					cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

				if ear < EYE_AR_THRESH:

					face_list[objectID]['consec_frame'] += 1

					if face_list[objectID]['consec_frame'] >= EYE_AR_CONSEC_FRAMES:
						cv2.putText(frame, '(EAR)Mengantuk!', (x - 10, y - 18), cv2.FONT_HERSHEY_SIMPLEX, 
							0.4, ((0, 0, 255)), 1)

						ear_e2 = cv2.getTickCount()
						ear_last_frame_exec_time = (ear_e2 - e1)/cv2.getTickFrequency()
			
						if exec_time_30_frame.full():
							exec_time_30_frame.get()
						exec_time_30_frame.put(ear_last_frame_exec_time)
						
						exec_time_list.append({
							'timestamp': millis_convert(vs_time),
							'id': objectID,
							'detector': 'ear',
							'time': list(exec_time_30_frame.queue)
						})

				else:
					face_list[objectID]['consec_frame'] = 0

				if mar > MOUTH_AR_THRESH:
					cv2.putText(frame, '(MAR)Mengantuk!', (x - 10, y - 18), cv2.FONT_HERSHEY_SIMPLEX, 
						0.4, (255, 0, 0), 1)

					mar_e2 = cv2.getTickCount()
					mar_exec_time = (mar_e2 - e1)/cv2.getTickFrequency()

					exec_time_list.append({
						'timestamp': millis_convert(vs_time),
						'id': objectID,
						'detector': 'mar',
						'time': mar_exec_time
					})

			else:
				continue

	cv2.namedWindow('Deteksi Kantuk', cv2.WINDOW_NORMAL) 
	cv2.imshow('Deteksi Kantuk', frame)
	key = cv2.waitKey(1) & 0xFF
	fps.update()

	e2 = cv2.getTickCount()
	exec_time = (e2 - e1)/cv2.getTickFrequency()
	
	if exec_time_30_frame.full():
		exec_time_30_frame.get()
	exec_time_30_frame.put(exec_time)

	if key == ord("q"):
		# generate_csv(exec_time_list)
		break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.release()

# py multi_drowsiness.py -v ../dataset_to_used/dataset_variation/eight_720.mp4