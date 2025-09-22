from util.model import RapidOcr,AnchorModel,AnchorModelWithoutMonitor,RapidOcrGPU
from util.video import doOcr,getOcrTxts,getOcrResByTxt
import cv2
import pyautogui
import time
import random
import logging
import numpy as np

def displayOcrRes(img,txts,boxes):
    for box,txt in zip(boxes,txts):
        x1 = int(box[0][0])
        y1 = int(box[0][1])
        x2 = int(box[2][0])
        y2 = int(box[2][1])
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(img,txt,(x1,y1-35),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    return img

class YysStatusModel:
    def __init__(self):
        self.ocrModel = RapidOcr()
         # 为当前类创建专属日志器（推荐用类名作为日志器名称）
        self.logger = logging.getLogger(self.__class__.__name__)
        # 可根据需要单独设置日志级别（不设置则继承全局配置）
        self.logger.setLevel(logging.DEBUG)
        
        # （可选）为当前类的日志器添加独立处理器（如单独输出到文件）
        if not self.logger.handlers:  # 避免重复添加处理器
            # 控制台处理器
            console_handler = logging.StreamHandler()
            # 格式化器
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def oneStep(self,img):
        res = self.ocrModel.doOcr(img)
        txts,boxes = getOcrResByTxt(res,["挑战"])
        return txts,boxes

    def twoStep(self, boxes):
        """
        返回boxes中第一个box包围的随机一个坐标
        
        参数:
            boxes: 边界框列表，每个边界框为包含4个顶点的数组，
                每个顶点格式为[x, y]，如[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        
        返回:
            tuple: 随机坐标(x, y)，位于第一个边界框内部
            False: 若boxes为空或格式不正确
        """
        
        # 检查boxes是否为空
        if not boxes:
            self.logger.warning("two step error not find box! boxes: %s", boxes)
            return False,None
        
        # 取出第一个边界框
        box = boxes[0]
        
        # 检查边界框是否包含4个顶点
        if len(box) != 4:
            self.logger.warning("two step error box len error! box: %s", box)
            return False,None
        
        # 提取所有x和y坐标
        x_coords = [point[0] for point in box]
        y_coords = [point[1] for point in box]
        
        # 计算x和y的最小值与最大值（确定矩形范围）
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)
        
        # 检查坐标有效性
        if min_x >= max_x or min_y >= max_y:
            self.logger.warning("two step error box range error! box: %s", box)
            return False,None
        
        # 在矩形范围内生成随机坐标
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        
        return True,(x, y)

    def threeStep(self, x, y):
        """
        鼠标点击指定坐标(x, y)的位置
        
        参数:
            x (float/int): 要点击的x坐标
            y (float/int): 要点击的y坐标
        
        返回:
            bool: 操作成功返回True，失败返回False
        """
        try:
             # 检查坐标是否为有效数字（兼容Python原生类型和numpy数字类型）
            def is_number(n):
                return isinstance(n, (int, float, np.number))  # 增加numpy数字类型判断
            
            if not (is_number(x) and is_number(y)):
                raise ValueError(f"坐标必须是数字类型，实际x类型: {type(x)}, y类型: {type(y)}")
            
            # 转换为Python原生float，再取整（避免numpy类型带来的潜在问题）
            click_x = int(round(float(x)))
            click_y = int(round(float(y)))
            
            # 获取屏幕尺寸，检查坐标是否在屏幕范围内
            screen_width, screen_height = pyautogui.size()
            if not (0 <= click_x < screen_width and 0 <= click_y < screen_height):
                raise ValueError(f"坐标({click_x}, {click_y})超出屏幕范围，屏幕尺寸为({screen_width}, {screen_height})")
            
            # 生成随机延迟（例如0.3到1.2秒之间），可根据需要调整范围
            random_delay = random.uniform(0.1, 0.3)
            time.sleep(random_delay)
            
            # 执行点击操作
            pyautogui.click(click_x, click_y)
        
            # 可选：添加短暂延迟，确保点击操作完成
            time.sleep(0.1)
            
            return True
        
        except Exception as e:
            self.logger.error("three step error! x: %s, y: %s, error: %s", x, y, str(e))
            return False

    def step(self,img):
        _,boxes = self.oneStep(img)
        if len(boxes) == 0:
            self.logger.warning("one step error not find! boxes: %s", boxes)
            return False
        ret,pos = self.twoStep(boxes)
        if not ret:
            self.logger.warning("two step error!")
            return False
        ret = self.threeStep(*pos)
        if not ret:
            self.logger.warning("three step error!")
            return False
        return  True

if __name__ == "__main__":
    yysStatusModel = YysStatusModel()
    img = cv2.imread("./pic/1.png")
    yysStatusModel.step(img)

