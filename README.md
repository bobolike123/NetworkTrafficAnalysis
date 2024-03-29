


# NetworkTrafficAnalysis 网络流量分析代码
* 论文:A Deep Hierarchical Network for Packet-Level Malicious Traffic Detection的源代码

* 模型训练代码:项目根目录下:CICIDS2017_6class.py、ISCX2012_LSTM_5class.py、USTC-TFC2016_20class.py
* 模型预测代码:ModelPredict.py
* 所需环境：python3.6+CUDA+CUDnn+根目录下requirements.txt(注意keras和tensorflow-gpu的版本号要匹配)
* 重要项目结构说明:
```text
--NetworkTrafficAnalysis--
    --CICIDS2017_6class.py
    --ISCX2012_LSTM_5class.py
    --USTC-TFC2016_20class.py
    --ModelPredict.py
    --requirements.txt
    ----checkpoints  #保存每轮训练的权重 （需自创）
    ----log #保存每次训练的日志记录
    ----preprocess #预处理代码
    ----testcode #测试代码，可实现自动化多次训练
    ----tools #可用到的工具代码
    ----evaluation #存放训练过程中的评估数据的文件夹（需自创）
```
## 注意
* 由于源数据集达数十个G，不方便上传，下面是数据库的下载链接。项目内有预处理代码，感兴趣的可以研究。
* 1、 www.unb.ca/cic/datasets （CICIDS2017 and ISCXIDS2012）
* 2、 github.com/yungshenglu/USTC-TFC2016 (USTCTFC2016)
* 此外，源代码包括处理后的数据集，不包括权重文件。请通过数据集(dataset文件夹内)进行生成，生成的权重文件在checkpoints文件夹内,训练过程中的评估数据放在evaluation文件夹内
* **由于github上传不了空文件夹，checkpoints 和evaluation文件夹需要手动创建,文件夹位置就在根目录下**

## 问题反馈
在使用中有任何问题，欢迎反馈给我，可以用以下联系方式跟我交流
* 邮件(zhudunpap@163.com)

## 感激
如果觉得这篇文献和代码对您有帮助，希望能给这个项目点个star，同时也希望得到您的引用，引用格式在下面

## 处理后的数据集
应广大学者要求，现放出预处理过后神经网络训练用的数据集。前几步生成的中间文件因为生成的小文件过多不方便打包，若希望复现请请自行尝试。
该数据集仅供参考，不代表论文中所用的数据集，因处理不当导致误差较大的，作者不对此负责。数据集有时间限制，若过期请联系邮箱。
https://musetransfer.com/s/q6wyuw7v0（有效期至2023年4月8日）

## 文献引用
```text
引用格式：B. Wang, Y. Su, M. Zhang and J. Nie, "A Deep Hierarchical Network for Packet-Level Malicious Traffic Detection,"
in IEEE Access, vol. 8, pp. 201728-201740, 2020, doi: 10.1109/ACCESS.2020.3035967.
```
