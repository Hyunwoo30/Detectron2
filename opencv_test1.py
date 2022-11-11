from pickletools import uint8
import cv2
from cv2 import LINE_AA
from cv2 import LINE_8
import numpy as np
import matplotlib.pyplot as plt

src = cv2.imread("./grabImg_0_221025_142513.png")
image = cv2.resize(src, (700, 700))
height, width, channel = image.shape
# print(height, width, channel)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = hsv[:, :]
edges = cv2.Canny(gray,500,600)


img_blurred = cv2.GaussianBlur(edges, ksize=(7, 7), sigmaX=0)

img_blur_thresh = cv2.adaptiveThreshold(
    img_blurred,
    maxValue=255.0,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=17,
    C=7
)
contours, _ = cv2.findContours(
    img_blur_thresh,
    mode=cv2.RETR_CCOMP, # 모든 윤곽선을 검출하며, 계층 구조는 2단계로 구성합니다.
    method=cv2.CHAIN_APPROX_SIMPLE # 윤곽점들 단순화 수평, 수직 및 대각선 요소를 압축하고 끝점만 남겨 둡니다.
)

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255,255,255))

# temp_result = np.zeros((height, width, channel), dtype=np.uint8)

contours_dict = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(temp_result, pt1=(x,y), pt2=(x+w, y+h), color=(0,0,255), thickness=2)
    
    contours_dict.append({
        'contour': contour,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'cx': x + (w / 2),
        'cy': y + (h / 2),
        'h1': h - 30,
        # 'w1': w - 67
        'w1': (w//2)

    })
cv2.imshow("draw", contours_dict)
cv2.waitKey()
cv2.destroyAllWindows()
MIN_AREA = 50
MIN_WIDTH, MIN_HEIGHT=60, 70
MIN_RATIO, MAX_RATIO = 0.25, 1.0

possible_contours = []

cnt = 0
for d in contours_dict:
    area = d['w'] * d['h']
    ratio = d['w'] / d['h']
    
    # cx = d['cx']
    # cy = d['cy']
    if area > MIN_AREA \
    and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
    and MIN_RATIO < ratio < MAX_RATIO:
        d['idx'] = cnt
        cnt += 1
        possible_contours.append(d)

temp_result = np.zeros((height, width, channel), dtype = np.uint8)

for d in possible_contours:
    cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(0, 255, 0), thickness=2)
    cv2.line(temp_result, (d['x']+d['w1'], d['x']), (d['x']+d['w1'], d['y']+d['h1']), (0, 0, 255), thickness=LINE_8)

final = cv2.bitwise_or(image, temp_result)


# cv2.imshow("draw", final)
# cv2.waitKey()
# cv2.destroyAllWindows()













# ret, binary = cv2.threshold(imgray, 252, 255, cv2.THRESH_BINARY_INV)
# binary = cv2.bitwise_not(binary)
# contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
# image, contour = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
# dst = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=9)


# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5), anchor=(-1, -1))
# dst = cv2.erode(src, kernel, iterations=2)
# merged = np.hstack((image, binary))


# plt.figure(figsize=(15, 20))
# plt.imshow(gray)
# plt.show()