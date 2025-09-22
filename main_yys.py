from util.model import RapidOcr,AnchorModel,AnchorModelWithoutMonitor,RapidOcrGPU
from util.video import doOcr,getOcrTxts,getOcrResByTxt
import cv2
import pyautogui
import time
import random
import logging
import numpy as np
from datetime import datetime
import argparse

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
    def __init__(self,use_cuda=False):
        if use_cuda:
            self.ocrModel = RapidOcrGPU()
        else:
            self.ocrModel = RapidOcr()
        self.use_cuda = use_cuda
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

    def oneStep(self,img,txts):
        res = self.ocrModel.doOcr(img)
        txts,boxes = getOcrResByTxt(res,txts,use_cuda=self.use_cuda)
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
        txts = ["挑战","点击屏幕继续"]
        ocr_txts,boxes = self.oneStep(img,txts)
        self.logger.info("one step find txts: %s", ocr_txts)
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
        self.logger.info("click success! pos: %s", pos)
        return  True

    def __call__(self, img):
        time_start = time.time()
        _ = self.step(img)
        time_end = time.time()
        self.logger.info("step cost time: {:.2f}ms".format((time_end - time_start)*1000))

def capture_screenshot(save_pic = False,save_path=None):
    """
    在 macOS 上截取屏幕并保存
    
    参数:
        save_path: 保存路径，默认为当前目录，文件名含时间戳
    """
    try:
        # 生成带时间戳的文件名（避免重复）
        if not save_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"screenshot_{timestamp}.png"
        
        # 截取全屏
        screenshot = pyautogui.screenshot()
        
        # 保存图片
        if save_pic:
            screenshot.save(save_path)
            print(f"截屏成功，保存至: {save_path}")

        # 转换为NumPy数组（形状为 (height, width, 3)，RGB）
        rgb_img = np.array(screenshot)
        
        # 转换通道顺序：RGB → BGR（cv2默认使用BGR）
        cv2_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        return True,cv2_img
    
    except Exception as e:
        print(f"截屏失败: {str(e)}")
        return False,None


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process configuration and directories.")
    parser.add_argument('--use_cuda', type=bool, default=False,
                        help='Use CUDA for training.')
    args = parser.parse_args()
    return args

def test():
    txts = ["挑战","点击屏幕继续"]
    for obj in target:
            # 每个目标的第二个元素是文字标签
            txt = obj
            flag = True
            for t in txts:
                if t in txt:
                    flag = False
                    break
            if flag:
                continue
            txts.append(txt)

if __name__ == "__main__":
    # 示例：延迟 3 秒后截屏（给切换窗口留时间）
    args = parse_arguments()
    yysStatusModel = YysStatusModel(args.use_cuda)

    while(True):
        ret,img = capture_screenshot()
        if not ret:
            print("capture screenshot error")
            exit(-1)

        ret = yysStatusModel(img)

