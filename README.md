
## 人脸识别系统
项目中的Faster-RCNN检测算法， 参考的[keras-frcnn][1]

该项目是主要用于训练，其部署机器api接口，移步[这里][4]

该系统主要包含三个模型：

1. 人脸分类 face_classifier

   人脸分类模型预期是判别图像是否是人脸，其主要作用是验证简化版的VGG模型能否准确对人脸进行分类，其模型架构将作为人脸定位模型的Classfier。

   简化改进：
    1. 在卷积层中，保留了前4个Block，并每次卷积与激活函数之间加入了BatchNormalization。去掉了第5个Block， 这里把前4个Block称为BaseLayer;
    1. 全连接层中，减少了两个全连接层的数量。

   训练数据：
    1. 正样本：按照VGG-Face-Dataset中annotations中的bundling box标记位置，截取人脸。
    1. 负样本：bundling box 上下左右的随机图像。
    样本数：训练集：145177， 开发集：12249， 测试集：12249

   输入图像：[64, 64]

   test set:

   num_pos: 5000

       num_neg: 7243

       loss = 0.006139507154439343

       accuracy = 0.99796875
2. 人脸定位 Faster-RCNN

    我的模型训练是在Faster-RCNN的VGG16版本进行了简化修改。其BaseLayer复用人脸分类的BaseLayer，其classfier结构与人脸分类模型一致， rpn 则结构不变。

    训练数据集：对VGG-Face-Dataset中annotations中的图像进行过滤。
    1. Vgg Face Dataset 数据集中一张图像对应一个人脸标记，在frcnn学习中，会把标记之外的，划分为负样本(bg); 因此得去掉存在多个人脸的图像
    2. 部分url失效，或者url下载到的图像并非原有图像，或者存在标记错误。

    样本总数：64421 (只选择了部分图像，源VGG-Face数据集中有200w+的图像。)

3. 人脸识别

    模型参考了[Deep Face Recognition][2], 策略是先训练一个输出为2622的CNN分类器(VGG)，在使用阶段，去掉Top-Layer，这样模型的输出即为人脸的编码，再通过计算不同人脸的欧氏距离，即可判断人脸相似度，从而识别人脸。原论文中，还模型输出后，加了triplet loss层，可大幅提升人脸识别准确率。这里并没有加triplet loss层。

    训练数据集：
    1. 与人脸定位的原图像相同，只是把每张图像的人脸提取出来，按不同人名不同文件夹的方式保存，总共2622个类别。
    1. 考虑到人脸定位阶段给出的bbox位置可能，有误差，所以这里以annotation中给的bbox的中心位置，进行随机裁剪，进行数据增强。还可以做水平翻转，能提高鲁棒性(不过我的计算资源有限，就没做了。)


    样本总数：64421 * 2

    不过这里的样本总数明显远远不够，因为总类别数达到了2622. 所以这个模型的训练没有达到好的效果。


[1]:https://github.com/yhenon/keras-frcnn "keras-frcnn"
[2]:http://www.robots.ox.ac.uk/~vgg/software/vgg_face/ "vgg-face-dataset"
[3]:http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf "Deep Face Recognition"
[4]:https://github.com/zc-play/rpc-server "rpc-server"
