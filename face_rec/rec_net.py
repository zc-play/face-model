# coding: utf-8
import os
from keras.models import Model, Input
from keras.optimizers import Adam
from keras.layers import Dense, Flatten, Conv2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, History

from face_rec.config import Config
from face_classifier.net import nn_base


cfg = Config()
epochs = 2000


def get_rec_model(pre_train_path=None, is_train=False):
    img_input = Input(shape=(64, 64, 3))
    x = nn_base(img_input, trainable=True)
    # decrease parameters numbers
    x = Conv2D(4096, (1, 1), activation='relu')(x)
    x = AveragePooling2D((4, 4), name='avg_pool')(x)
    x = Flatten(name='encoding_out')(x)
    # dense output
    x = Dense(cfg.class_size, activation='softmax', name='fc2622')(x)
    model = Model(inputs=img_input, outputs=x)
    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.input_shape, 'trainable:{}'.format(layer.trainable))

    optimizer = Adam(lr=1e-5)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    if is_train and pre_train_path:
        model.load_weights(pre_train_path, by_name=True)
    return model


def train(train_path, val_path):
    pre_train_path = cfg.pre_train_model_path
    if os.path.exists(cfg.model_path):
        while True:
            is_import = input('The model already exists, is import it and continue to train(Y) or overwrite it(N): ')
            if is_import.lower() == 'y':
                pre_train_path = cfg.model_path
                break
            elif is_import.lower() == 'n':
                break
    model = get_rec_model(pre_train_path, is_train=True)
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True
    )
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(64, 64),
        batch_size=32,
        seed=5
    )

    val_datagen = ImageDataGenerator(
        rescale=1. / 255
    )
    val_generator = val_datagen.flow_from_directory(
        val_path,
        target_size=(64, 64),
        batch_size=32,
        seed=5
    )

    model_checkpoint = ModelCheckpoint(cfg.model_path, save_best_only=False)
    reduce_lr = ReduceLROnPlateau(patience=1)
    board = TensorBoard(cfg.train_log)
    hist = History()

    model.fit_generator(
        train_generator,
        steps_per_epoch=20,
        epochs=epochs,
        workers=8,
        use_multiprocessing=False,
        validation_data=None,
        callbacks=[model_checkpoint, reduce_lr, board, hist]
    )

    print('train completed!!!')


def save_model_no_top_weight():
    model = get_rec_model()
    model.load_weights(cfg.model_path)
    out = model.get_layer('encoding_out').output
    encoding_model = Model(model.input, out)
    # encoding_model.compile('adam')
    encoding_model.save(cfg.no_top_model_path)


def get_model_no_top():
    model = get_rec_model()
    model.load_weights(cfg.model_path)
    out = model.get_layer('encoding_out').output
    encoding_model = Model(model.input, out)
    encoding_model.compile('adam')
    encoding_model.load_weights(cfg.no_top_model_path)
    return encoding_model


if __name__ == '__main__':
    # env PYTHONPATH='..' python rec_net.py
    train(cfg.train_path, cfg.dev_path)
    # save_model_no_top_weight()
