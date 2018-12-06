# coding:utf8
import os

import cv2


def detect():
    # 创建人脸检测的对象
    face_cascade = cv2.CascadeClassifier("./venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default"
                                         ".xml")
    # 连接摄像头的对象 0表示摄像头的编号
    camera = cv2.VideoCapture(0)
    j = 1
    k = 1
    while True:
        # 读取当前帧
        ret, frame = camera.read()
        # 转为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 检测人脸 返回列表 每个元素都是(x, y, w, h)表示矩形的左上角和宽高
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # 画出人脸的矩形
        for (x, y, w, h) in faces:
            # 画矩形 在frame图片上画， 传入左上角和右下角坐标 矩形颜色 和线条宽度
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # 把脸单独拿出来
            roi_gray = gray[y: y + h, x: x + w]
            img1 = frame[y: y + h, x: x + w]
            cv2.imwrite("./Yale/16/s" + str(k) + '.bmp', roi_gray)
            pic = cv2.imread("./Yale/16/s" + str(k) + '.bmp')
            pic = cv2.resize(pic, (100, 100), interpolation=cv2.INTER_CUBIC)
            os.remove("./Yale/16/s" + str(k) + '.bmp')
            if j % 20 == 0:
                cv2.imwrite("./Yale/16/s" + str(k) + '.bmp', pic)
                k += 1
                print(str(int(j / 20)))
            j = j + 1

            # 在脸上检测眼睛   (40, 40)是设置最小尺寸，再小的部分会不检测
            # eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, 0, (40, 40))
            # 把眼睛画出来
            # for(ex, ey, ew, eh) in eyes:
            #     cv2.rectangle(img, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
        if j == 240:
            break
        cv2.imshow("camera", frame)
        if cv2.waitKey(5) & 0xff == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect()
