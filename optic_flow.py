import cv2
import numpy as np

def select_new_roi(frame):
    roi = cv2.selectROI("Виберіть об'єкт для відстежування", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Виберіть об'єкт для відстежування")
    
    if roi != (0, 0, 0, 0):
        return roi
    else:
        return None

cap = cv2.VideoCapture('fpv2.mp4')

ret, first_frame = cap.read()
if not ret:
    print("Не вдалося завантажити відео.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

roi = select_new_roi(first_frame)
if roi is None:
    print("Не вибрано жодного об'єкта.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

x, y, w, h = roi
old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray[y:y+h, x:x+w], mask=None, maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)
if p0 is not None:
    p0[:, :, 0] += x
    p0[:, :, 1] += y
else:
    print("Не вдалося знайти достатню кількість точок для відстеження.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        if len(good_new) > 0:
            dx = np.mean(good_new[:, 0] - good_old[:, 0])
            dy = np.mean(good_new[:, 1] - good_old[:, 1])
            x += int(dx)
            y += int(dy)

        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow('frame', frame)

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        
        if len(good_new) < 10:  # Якщо кількість точок занадто мала, оновити точки
            p0 = cv2.goodFeaturesToTrack(frame_gray[y:y+h, x:x+w], mask=None, maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)
            if p0 is not None:
                p0[:, :, 0] += x
                p0[:, :, 1] += y
            else:
                print("Не вдалося знайти достатню кількість точок для відстеження.")

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('r'):
        roi = select_new_roi(frame)
        if roi is not None:
            x, y, w, h = roi
            p0 = cv2.goodFeaturesToTrack(frame_gray[y:y+h, x:x+w], mask=None, maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)
            if p0 is not None:
                p0[:, :, 0] += x
                p0[:, :, 1] += y
                old_gray = frame_gray.copy()
            else:
                print("Не вдалося знайти достатню кількість точок для відстеження.")

cap.release()
cv2.destroyAllWindows()
