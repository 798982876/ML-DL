# 人脸检测

[参考文档](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/)
## 前期准备
### 环境安装

````python
#创建新的虚拟环境
conda create -n face python=3.7
#使用镜像安装tensorflow
pip install tensorflow==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
#其他包安装
pip install sklearn
pip install imutils
pip install matplotlib

````
### 照片整理
训练集——dataset 分为with_mask,without_mask文件夹

## 训练
### 第一次
````python
#超参数
INIT_LR = 1e-4                          #the initial learning rate
EPOCHS = 20                             #number of epochs to train for
BS = 32                                 #batch size
````
训练评价
<img src="assets\训练模型评价.png">
<img src="assets\plot.png">
**过拟合**

### 第二次 调整超参数
````python
#超参数
INIT_LR = 1e-4                          #the initial learning rate
EPOCHS = 10                             #number of epochs to train for
BS = 32                                 #batch size
````
训练评价
<img src="assets\训练模型评价2.png">
<img src="assets\plot10.png">
**过拟合**

### 第三次 调整超参数
````python
#超参数
INIT_LR = 1e-4                          #the initial learning rate
EPOCHS = 7                             #number of epochs to train for
BS = 32                                 #batch size
````
训练评价
<img src="assets\训练模型评价3.png">
<img src="assets\plot7.png">
**OK**