import time

import cv2 as cv

from gym_flexassembly.vision.pose_detection.projection.side_model import SidePredictor

prediction_time = 0

img_hole = cv.imread('./datasets/side/val/hole/0045.png')
img_other = cv.imread('./datasets/side/val/other/0045.png')
side_predictor = SidePredictor('models/side_model.pth')

since = time.time()
pred = side_predictor.predict(img_hole)
prediction_time += time.time() - since
cv.imshow(f'Hole: {pred == 0}', img_hole)

since = time.time()
pred = side_predictor.predict(img_other)
prediction_time += time.time() - since
cv.imshow(f'Other: {pred == 1}', img_other)

print(f'Avg. prediction time: {(prediction_time / 2) * 1000:.3f}ms')

cv.waitKey(0)
