# NetworkTrafficAnalysis
论文:A Deep Hierarchical Network for Packet-Level Malicious Traffic Detection的源代码

模型训练代码:项目根目录下:CICIDS2017_6class.py、ISCX2012_LSTM_5class.py、USTC-TFC2016_20class.py
模型预测代码:ModelPredict.py
所需环境：python3.6+CUDA+CUDnn+根目录下requirements.txt(注意keras和tensorflow-gpu的版本号要匹配)
重要项目结构说明:
--NetworkTrafficAnalysis--
    --CICIDS2017_6class.py
    --ISCX2012_LSTM_5class.py
    --USTC-TFC2016_20class.py
    --ModelPredict.py
    --requirements.txt
    ----checkpoints  #保存每轮训练的权重
    ----log #保存每次训练的日志记录
    ----preprocess #预处理代码
    ----testcode #测试代码，可实现自动化多次训练
    ----tools #可用到的工具代码

