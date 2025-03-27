# 用于校准黑色遮罩的RGB值为0,255
# 0,0,0表示黑色，255,255,255表示白色
import numpy as np
import cv2

if __name__ == '__main__':
    filename = '/data1/wjx/S003/input/black_mask/black_mask.png'
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    black = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    black = cv2.resize(black,(2004,1080))
    black = cv2.cvtColor(black, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(filename.replace('.png','_2004x1080.png'), black)