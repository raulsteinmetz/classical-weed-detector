import cv2

img1 = cv2.imread('1.png')
img2 = cv2.imread('2.png')

# resize using intercubic
res1 = cv2.resize(img1, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
res2 = cv2.resize(img2, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# save
cv2.imwrite('1_res.png', res1)
cv2.imwrite('2_res.png', res2)