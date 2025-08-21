import time

import cv2
import mss
from util.model import RapidOcr,AnchorModel
from util.anchor import getSquareParamC,preImageSolve,squareDetect
from util.draw import drawMatrix
import numpy as np
from util.video import videoSolve

def get_max_frame_rate_without_io():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        start_time = time.time()
        screenshot = sct.grab(monitor)
        cost_time = time.time() - start_time
    print(f"max frame rate without io: {1 / cost_time}")

def doSquare():
    
    ocrModel = RapidOcr()
    anchorModel = AnchorModel()

    while(True):
        img = anchorModel.getScreenShotWithOutVis()
        succ, resX, resY, resDistance = getSquareParamC(img, display=False)

        zero_matrix = np.zeros((6, 6), dtype=int)
        if not succ:
            print("未找到目标")
        else:
            preImage = preImageSolve(img)
            zero_matrix = squareDetect(preImage, resX, resY, resDistance, display=False)

        # 生成显示图像
        displayImage = drawMatrix(zero_matrix)
        cv2.imshow("displayImage",displayImage)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def Test():
    url = "./pic/4.png"
    ocrModel = RapidOcr()
    anchorModel = AnchorModel()

    img = cv2.imread(url)
    img,(lowH, lowW, highH, highW) = anchorModel.getAnchor(img)


    succ, resX, resY, resDistance = getSquareParamC(img, display=True)

    zero_matrix = np.zeros((6, 6), dtype=int)
    if not succ:
        print("未找到目标")
    else:
        preImage = preImageSolve(img)
        zero_matrix = squareDetect(preImage, resX, resY, resDistance, display=True)

    # 生成显示图像
    displayImage = drawMatrix(zero_matrix)

def test_gpu():

    import paddle
    from util.model import RapidOcrGPU
    print(paddle.utils.run_check())
    print(paddle.device.get_device())

    url_list = ["train_time_210s.jpg"]

    ocrModel = RapidOcrGPU()
    anchorModel = AnchorModel()    

    img_list = []
    for url in url_list:
        img = cv2.imread(url)
        img_list.append(img)


    img_tmp,(lowH, lowW, highH, highW) = anchorModel.getAnchor(img_list[0])
    res = ocrModel.doOcr(img_tmp)

    print(res)

    return 
    start_time = time.time()
    count = 0
    while(True):
        index = count%len(img_list)
        img_tmp,(lowH, lowW, highH, highW) = anchorModel.getAnchor(img_list[index])
        res = ocrModel.doOcr(img_tmp)

        count += 1
        if count % 10 == 0:
            end_time = time.time()
            print("{:.2f}ms".format((end_time - start_time)*1000/10))
            start_time = end_time


def test_cpu():
    ocrModel = RapidOcr()
    anchorModel = AnchorModel()
    anchorModel.changeRate(0.69, 0.75, 0.08, 0.20)

    img =cv2.imread("train_time_210s.jpg")

    img_anchor,(lowH, lowW, highH, highW) = anchorModel.getAnchor(img)

    cv2.imshow("img",img_anchor)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 帮我计算100次平均时间
    start_time = time.time()
    for i in range(10):
        res = ocrModel.doOcr(img_anchor)
    end_time = time.time()
    print("10次平均时间: {:.2f}ms".format((end_time - start_time)*1000/10))

 
def doOcrContinue():
    ocrModel = RapidOcr()
    anchorModel = AnchorModel()
    # 获取截图
    while(True):
        start_time = time.time()
        img = anchorModel.getScreenShot()
        ocrRes = ocrModel.displayImage(img)
        ela_time = time.time() - start_time
        ela_time = int(ela_time*1000)

        str = "{}ms".format(ela_time)

        ocrRes = ocrModel.addChineseText(ocrRes,str,(40,35))

        anchorModel.disPlayImage(ocrRes)
        time.sleep(0.1)

def doOcrV1():
    ocrModel = RapidOcr()
    anchorModel = AnchorModel()

    # 计算帧率60
    start_time = time.time()
    frame_count = 0
    frame_rate = 40.0
    bound_time = 1/frame_rate

    while(True):
        begin = time.time()
        img = anchorModel.getScreenShotWithOutVis()

        # 计算当前时间
        current_time = time.time()
        # 计算已经运行的时间
        elapsed_time = current_time - begin

        # 更新开始时间
        if elapsed_time < bound_time:
            time.sleep(bound_time-elapsed_time)

        # 计算已经运行的时间
        elapsed_time = time.time() - start_time
        # 更新开始时间

        frame_count += 1
        if frame_count % 30 == 0:
            print("平均帧率: {:.2f}fps".format(frame_count / elapsed_time))
            
            start_time = current_time
            frame_count = 0



if __name__ == "__main__":
    from util.video import videoSolve,imageFilter,doFilter
    from util.createDataset import doCreateDataset,draw

    from util.stateMachine import videoSolver,offLineVideoSolver

    from util.multi_video import multiVideoTask

    # videoSolve(use_cuda=True)
    # multiVideoTask()
    # imageFilter("./dataset")
    # doCreateDataset()
    # doOcrV1()
    videoSolver = offLineVideoSolver(use_cuda=False)
    videoSolver.mainSolve("./video/train.mp4")
    # test_gpu()
    # img = cv2.imread("10562.jpg")
    # suc,box =doFilter(img)
    # if suc:
    #     cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)




