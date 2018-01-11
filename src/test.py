from fileUtil import FileUtil
from evalUtil import EvalUtil
from visualize import VisualUtil


if __name__ == '__main__':
    fUtil = FileUtil()
    eUtil = EvalUtil()
    vUtil = VisualUtil()
    fileList, gains = fUtil.getResults()

    for index in range(len(fileList)):
        print '%s   %s'%(fileList[index],eUtil.evalGain(gains[index]))
    
    vUtil.drawMultiTest(gains)


