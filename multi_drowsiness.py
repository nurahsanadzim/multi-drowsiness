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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
ap.add_argument("-v", "--video", required=True, help="path to input video file")
args = vars(ap.parse_args())
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the alarm
EYE_AR_THRESH = 0.15
EYE_AR_CONSEC_FRAMES = 48

# define one constants, for mouth aspect ratio to indicate open mouth
MOUTH_AR_THRESH = 0.83

# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0

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

# input file video
vs = cv2.VideoCapture(args["video"])
fps = FPS().start()

face_list = []

debug_count = 0
# loop over frames from the video stream
while True:

	if debug_count != 0:
		break

	# grab the frame from the threaded video file stream
	(grabbed, frame) = vs.read()
	
	# if the frame was not grabbed, then we have reached the end of the stream
	if not grabbed:
		break

	# konversi frame ke grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)
		# index untuk pemisah setiap objek wajah yang terdeteksi
	# face_count = 0

	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# (x, y, w, h) = face_utils.rect_to_bb(rect)
		# print("{}, {}, {}, {}".format(x, y, w, h))
		print(rect.tl_corner())
		print(rect.tr_corner())
		print(rect.bl_corner())
		print(rect.br_corner())
		print("----------")

		# cek jika list wajah belum ada sama sekali
		try:
			face_list[i]
		except IndexError:
			face_list.append({
				'consec_frame': 0,
				'ear_time': queue.Queue(maxsize=30),
				'mar_time': 0
			})


		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# mulai hitung waktu eksekusi untuk EAR
		e1 = cv2.getTickCount()

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

			# naikkan consec_frame jika EAR melewati nilai ambang
			face_list[i]['consec_frame'] += 1

			# if the eyes were closed for a sufficient number of
			# then sound the alarm
			# if COUNTER >= EYE_AR_CONSEC_FRAMES:
			if face_list[i]['consec_frame'] >= EYE_AR_CONSEC_FRAMES:
				print("wajah ke {} mengantuk dari EAR".format(i))
				# simpan waktu eksekusi
		
		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the counter
		else:
			face_list[i]['consec_frame'] = 0

		# selesai hitung waktu eksekusi EAR
		e2 = cv2.getTickCount()

		(x, y, w, h) = face_utils.rect_to_bb(rect)
		cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		# cv2.putText(frame, "EAR: {:.2f}".format(ear), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# mulai hitung waktu eksekusi untuk MAR
		e3 = cv2.getTickCount()

		# extract the mouth coordinates, then use the
		# coordinates to compute the mouth aspect ratio
		mouth = shape[mStart:mEnd]

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# hitung MAR
		mouthMAR = mouth_aspect_ratio(mouth)
		mar = mouthMAR

		# compute the convex hull for the mouth, then
		# visualize the mouth
		mouthHull = cv2.convexHull(mouth)
		
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
		# cv2.putText(frame, "MAR: {:.2f}".format(mar), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# Draw text if mouth is open
		if mar > MOUTH_AR_THRESH:
			print("wajah ke {} mengantuk dari MAR".format(i))
		
		# selesai hitung waktu eksekusi MAR
		e4 = cv2.getTickCount()

		# input waktu eksekusi EAR
		if face_list[i]['ear_time'].full():
			face_list[i]['ear_time'].get()
		face_list[i]['ear_time'].put((e2 - e1)/cv2.getTickFrequency())

		# input waktu eksekusi MAR
		face_list[i]['mar_time'] = e4 - e3 / cv2.getTickFrequency()

	debug_count += 1

	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	fps.update()

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
# py multi_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --video <path>