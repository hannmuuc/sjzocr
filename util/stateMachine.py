
from util.anchor import squareDetect,preImageSolve,getImgByZeroMatrix,getSquareParamC,drawRectImage,createContours
from util.model import RapidOcr,AnchorModel,AnchorModelWithoutMonitor,RapidOcrGPU
from util.video import doOcr,getOcrTxts
from collections import deque
import time
import numpy as np
import cv2
import queue
import threading

class checkStatusModel():
    def __init__(self,use_cuda):
        # 0 未检查状态 1 已检查状态
        self.use_cuda = use_cuda

        self.priorCheck = 0
        self.status =0
        if self.use_cuda:
            self.ocrModel = RapidOcrGPU()
        else:
            self.ocrModel = RapidOcr()
        self.h = 6
        self.w = 6
        self.priorMartrix = np.zeros((self.h,self.w))
        self.resX = -1
        self.resY = -1
        self.resDistance = -1
        self.bidirectional_array = deque()
        self.index =0
    

    def doCheckWithStatusOne(self,img):
        if self.status != 1:
            return 1,False,None
        preImage = preImageSolve(img,8)
        res = squareDetect(preImage,self.resX,self.resY,self.resDistance)
        zero_matrix = np.zeros((self.h,self.w), dtype=int)
        flag = 0

        for i in range(self.h):
            for j in range(self.w):
                if self.priorMartrix[i][j] == 1 and res[i][j] == 0:
                    zero_matrix[i][j] = 1
                    self.priorMartrix[i][j] = 0
                    flag =1
        
        if flag == 0:
            return 1,True,None

        suc,boundImg =getImgByZeroMatrix(img,zero_matrix,self.resX,self.resY,self.resDistance)
        if suc==False:
            return 1,False,None

        res = self.ocrModel.doOcr(boundImg)
        txts = getOcrTxts(res,self.use_cuda)
        if len(txts) == 0:
            return 1,True,None
        return 1,True,txts  

    def doCheckWithStatusZero(self,img,display=False):

        if self.status != 0:
            return 0,False,None
        suc,x,y,distance = getSquareParamC(img,display=False,ocrModel=self.ocrModel,use_cuda=self.use_cuda)
        print(suc,x,y,distance)
        if suc == False:
            return 0,False,None
        self.resX = x
        self.resY = y
        self.resDistance = distance
        preImage = preImageSolve(img,8)
        res = squareDetect(preImage,self.resX,self.resY,self.resDistance)

        if display:
            contours = createContours(x,y,distance,6,6)
            img_draw = drawRectImage(img,contours,res)
            cv2.imwrite("preImage.jpg",preImage)
            cv2.imwrite("img_draw.jpg",img_draw)


        self.priorMartrix = res
        self.status = 1
        return 0,True,None

    def doCheckWithStatus(self,img):
        if self.status == 0:
            return self.doCheckWithStatusZero(img)
        else:
            return self.doCheckWithStatusOne(img)
    
    def doCheckQueue(self):
        start = time.time()
        while(len(self.bidirectional_array) > 0):
            img = self.bidirectional_array.popleft()
            tag,succ,res = self.doCheckWithStatus(img)
            if tag == 0 and succ == False:
                self.doQueueSkip(60)
                continue
            if succ == True and res != None:
                print(res)
        end = time.time()
        print(f"checkQueue cost {(end-start)*1000}ms")
    
    def doQueueClearRightBase(self):
        left, right = 0, len(self.bidirectional_array) - 1
        # 二分查找第一个不满足条件的位置
        while left < right:
            mid = (left + right) // 2
            if not self.doCheckIsOk(self.bidirectional_array[mid]):
                right = mid
            else:
                left = mid + 1
        return left

    def doQueueClearRight(self):
        """从右侧清理队列，移除第一个不满足条件及之后的所有元素"""
        if not self.bidirectional_array:
            return
        left = self.doQueueClearRightBase()
                
        if not self.doCheckIsOk(self.bidirectional_array[left]):
            while len(self.bidirectional_array) > left:
                self.bidirectional_array.pop()
        else:
            self.bidirectional_array.clear()

    def doQueueClearLeft(self):
        """从左侧清理队列，保留第一个满足条件及之后的所有元素"""
        if not self.bidirectional_array:
            return
            
        left, right = 0, len(self.bidirectional_array) - 1
        # 二分查找第一个满足条件的位置
        while left < right:
            mid = (left + right) // 2
            if self.doCheckIsOk(self.bidirectional_array[mid]):
                right = mid
            else:
                left = mid + 1
        
        if self.doCheckIsOk(self.bidirectional_array[left]):
            for _ in range(left):
                self.bidirectional_array.popleft()
        else:
            self.bidirectional_array.clear()

    def doQueueSkip(self,skipNum):
        while(len(self.bidirectional_array) > 0 and skipNum > 0):
            self.bidirectional_array.popleft()
            skipNum -= 1

    def doCheckIsOk(self, img):
        res = self.ocrModel.doOcr(img)
        txts = getOcrTxts(res,self.use_cuda)
        for txt in txts:
            if "搜索物资" in txt:
                return True
        return False

    def doCheck(self,img,boundLength):
        self.bidirectional_array.append(img)
        if len(self.bidirectional_array) < boundLength:
            return False,None

        check_res = self.doCheckIsOk(img)
        self.index += 1
        print(check_res)

        if self.priorCheck == 0:
            if not check_res:
                self.bidirectional_array.clear()
                self.status = 0
            else:
                self.doQueueClearLeft()
                self.doQueueSkip(boundLength//6)
                self.doCheckQueue()
                self.priorCheck = 1
        else:
            if not check_res:
                self.doQueueClearRight()
                self.doCheckQueue()
                self.status = 0
                self.priorCheck = 0
            else:
                self.doCheckQueue()



class videoSolver():
    def __init__(self):
        self.name = "videoSolver"
        self.image_queue = queue.Queue(maxsize=100)
        self.checkStatusModel = checkStatusModel(use_cuda=True)


    def process_images(self,statusModel):
        """处理队列中图片的线程函数"""
        while True:
            try:
                # 从队列获取图片，超时时间1秒，防止线程无限阻塞
                img = self.image_queue.get(timeout=1)
                
                # 在这里添加图片处理逻辑，例如OCR识别
                if img is not None:
                    # 示例：执行OCR识别
                    statusModel.doCheck(img,60)
                    # 可以在这里添加对识别结果的处理
                    
                # 标记任务完成
                self.image_queue.task_done()
            except queue.Empty:
                # 队列为空时继续循环等待
                continue
            except Exception as e:
                print(f"处理图片时发生错误: {e}")
                continue
    

    def mainSolve(self):
        anchor_model = AnchorModel()
        
        # 启动图片处理线程
        processing_thread = threading.Thread(
            target=self.process_images, 
            args=(self.checkStatusModel,),
            daemon=True  # 守护线程，主程序退出时自动结束
        )
        processing_thread.start()

        # 计算帧率相关参数
        num = 0
        start_time = time.time()
        frame_count = 0
        frame_rate = 80.0
        bound_time = 1 / frame_rate

        try:
            while True:
                num+=1
                begin = time.time()
                img = anchor_model.getScreenShotWithOutVis()
                
                # 检查队列是否有空间，如果满了则舍弃图片
                if not self.image_queue.full():
                    try:
                        # 非阻塞方式放入队列，防止阻塞主线程
                        self.image_queue.put(img, block=False)
                    except queue.Full:
                        # 理论上这里不会触发，因为上面已经检查过，但做双重保障
                        print("队列已满，舍弃图片")
                else:
                    print("队列已满，舍弃图片")

                # 计算当前耗时
                current_time = time.time()
                elapsed_time = current_time - begin

                # 控制帧率
                if elapsed_time < bound_time:
                    time.sleep(bound_time - elapsed_time)

                # 计算并定期打印平均帧率
                frame_count += 1
                if frame_count % 30 == 0:
                    current_elapsed = current_time - start_time
                    print(f"平均帧率: {frame_count / current_elapsed:.2f}fps")
                    print(f"当前队列大小: {self.image_queue.qsize()}")
                    # 重置计数
                    start_time = current_time
                    frame_count = 0
                    
        except KeyboardInterrupt:
            print("程序被用户中断")
        finally:
            # 等待队列中所有任务处理完成
            self.image_queue.join()
            print("所有图片处理完成，程序退出")



