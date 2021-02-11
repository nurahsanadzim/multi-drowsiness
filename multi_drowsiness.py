# import the necessary packages
import queue
from scipy.spatial import distance as dist
from imutils.video import VideoStream, FPS
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear

def mouth_aspect_ratio(mouth):
	# compute the euclidean distances between the two sets of
	# vertical mouth landmarks (x, y)-coordinates
	A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
	B = dist.euclidean(mouth[4], mouth[8]) # 53, 57

	# compute the euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates
	C = dist.euclidean(mouth[0], mouth[6]) # 49, 55

	# compute the mouth aspect ratio
	mar = (A + B) / (2.0 * C)

	# return the mouth aspect ratio
	return mar

# fungsi untuk menginput exec time ke queue
def input_time(time_list, sec):
  if time_list.full():
    time_list.get()
  time_list.put(sec)

# class Face:
# 	def __init__(self, consec_frame, exec_time):
# 		self.consec_frame = consec_frame
# 		self.exec_time = exec_time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
ap.add_argument("-v", "--video", required=True, help="path to input video file")
args = vars(ap.parse_args())
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.4
EYE_AR_CONSEC_FRAMES = 48

# define one constants, for mouth aspect ratio to indicate open mouth
MOUTH_AR_THRESH = 0.83

# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
ALARM_ON = False

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# grab the indexes of the facial landmarks for the mouth
(mStart, mEnd) = (49, 68)

# start the video stream thread
# print("[INFO] starting video stream thread...")
# vs = VideoStream(src=args["webcam"]).start().sleep(1.0)
vs = cv2.VideoCapture(args["video"])
fps = FPS().start()

# face_list = [{
# 	'consec_frame': 0,
# 	'exec_time': []
# }]

face_list = []

# loop over frames from the video stream
while True:
	# grab the frame from the threaded video file stream
	(grabbed, frame) = vs.read()
	
	# if the frame was not grabbed, then we have reached the end of the stream
	if not grabbed:
		break

	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	_, frame = vs.read()
	(w, h, c) = frame.shape
	#syntax: cv2.resize(img, (width, height))
	frame = cv2.resize(frame,(400, h))

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# inisialisasi jumlah deteksi wajah data dimensi kedua	

	face_count = 0

	# loop over the face detections
	for rect in rects:
		e1 = cv2.getTickCount()

		try:
			face_list[face_count]
		except IndexError:
			face_list.append({
				'consec_frame': 0,
				'exec_time': queue.Queue(maxsize=48)
			})


		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:

			# COUNTER += 1
			# print(COUNTER)
			face_list[face_count]['consec_frame'] += 1

			# if the eyes were closed for a sufficient number of
			# then sound the alarm
			# if COUNTER >= EYE_AR_CONSEC_FRAMES:
			if face_list[face_count]['consec_frame'] >= EYE_AR_CONSEC_FRAMES:

				# # if the alarm is not on, turn qit on
				# if not ALARM_ON:
				# 	ALARM_ON = True

				# draw an alarm on the frame
				cv2.putText(frame, "MENGANTUK!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter and alarm
		else:
			face_list[face_count]['consec_frame'] = 0
			ALARM_ON = False

		# draw the computed eye aspect ratio on the frame to help
		# with debugging and setting the correct eye aspect ratio
		# thresholds and frame counters
		# cv2.putText(frame, "EAR: {:.2f}: pada wajah {}".format(ear, face_count), (0, 30*face_count),
		# 	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		cv2.putText(frame, "EAR: {:.2f}".format(ear), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# mycode
		# cv2.putText(frame, "Wajah: {}".format(face), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))

		# extract the mouth coordinates, then use the
		# coordinates to compute the mouth aspect ratio
		mouth = shape[mStart:mEnd]

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		mouthMAR = mouth_aspect_ratio(mouth)
		mar = mouthMAR
		# compute the convex hull for the mouth, then
		# visualize the mouth
		mouthHull = cv2.convexHull(mouth)
		
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
		# cv2.putText(frame, "MAR: {:.2f}".format(mar), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# Draw text if mouth is open
		if mar > MOUTH_AR_THRESH:
			cv2.putText(frame, "Mouth is Open!", (30,60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
		e2 = cv2.getTickCount()
		# print((e2 - e1)/cv2.getTickFrequency())

		# print(type(face_list[face_list]['exec_time']))
		if face_list[face_count]['exec_time'].full():
			face_list[face_count]['exec_time'].get()
		face_list[face_count]['exec_time'].put((e2 - e1)/cv2.getTickFrequency())			
		# input_time(, (e2 - e1)/cv2.getTickFrequency())
		
		print(face_list[face_count]['exec_time'].queue)
		# naikkan counter untuk deteksi selanjutnya
		face_count += 1

	print(face_list)

	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	fps.update()

	# print(face_list)

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()

# run the code
# py multi_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --video 10.MOV