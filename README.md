项目中的Faster-RCNN检测算法， 参考的[keras-frcnn][1]

人脸识别系统总共训练了3个模型：
    (1) 人脸分类 face_classifier
        模型训练：
        数据来源：原图像为vgg_face_dataset,
        输入图像：[64, 64]
        test set:
            num_pos: 5000
            num_neg: 7243
        loss = 0.006139507154439343
        accuracy = 0.99796875
    (2) 人脸定位 frcnn
    (3) 人脸识别


[1]:https://github.com/yhenon/keras-frcnn "keras-frcnn"
