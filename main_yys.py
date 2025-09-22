
from util.model import RapidOcr,AnchorModel,AnchorModelWithoutMonitor,RapidOcrGPU
from util.video import doOcr,getOcrTxts


if __name__ == "__main__":
    img =cv2.imread("./pic/1.png")
    ocrModel = RapidOcr()
    res = ocrModel.doOcr(img)
    txts = getOcrTxts(res,False)
    print(txts)

