from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2

import matplotlib.pyplot as plt
from matplotlib import ticker

IMAGE_SIZE = 256
IMAGE_SIZE_s = 128
TRAIN_NUM = 10000
VALID_NUM = 2000
TEST_NUM = 10000
EPOCH = 10
BATCH_SIZE = 50
LEARNING_RATE = 0.0005
MODEL_PATH = './1/model'


def readData():
    DIR_left = './1/train/left_process/'
    DIR_right = './1/train/right_process/'
    train_left = [DIR_left + i for i in os.listdir(DIR_left)]
    train_left.sort(key=lambda x: int(x.split('/')[4].split('.')[0].split('_')[0]))
    train_right = [DIR_right + i for i in os.listdir(DIR_right)]
    train_right.sort(key=lambda x: int(x.split('/')[4].split('.')[0].split('_')[0]))
    train_all = train_left + train_right
    train_processed = np.ndarray((TRAIN_NUM, IMAGE_SIZE_s, IMAGE_SIZE_s, 1), dtype=np.float32)
    for i in range(TRAIN_NUM):
        # directly load processed image
        train_processed[i, :, :, 0] = cv2.cvtColor(cv2.imread(train_all[i]), cv2.COLOR_BGR2GRAY)
        # train_processed[i, :, :, 0] = imagepre(train_all[i])
    test_processed = np.ndarray((TEST_NUM, IMAGE_SIZE_s, IMAGE_SIZE_s, 1), dtype=np.float32)
    '''
    DIR_test = './1/valid/'
    test_all = [DIR_test + i for i in os.listdir(DIR_test)]
    test_all = test_all[1:]
    for i in range(TEST_NUM):
        test_processed[i, 0, :, :] = imagepre(test_all[i])
    '''
    return train_processed, test_processed, train_all


# randomly shuffle the data
def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


# remove irrelevant background to get compact square shape image
def removeBorder(img):
    threshold = 5
    white = 240
    le = 0
    while (img[:, le] < white).sum() < threshold:
        le += 1
    ri = img.shape[1] - 1
    while (img[:, ri] < white).sum() < threshold:
        ri -= 1
    up = 0
    while (img[up, :] < white).sum() < threshold:
        up += 1
    do = img.shape[0] - 1
    while (img[do, :] < white).sum() < threshold:
        do -= 1
    return img[up:do, le:ri]


def imagepre(picture):
    img = cv2.imread(picture)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # remove white border to get compact image
    img = removeBorder(img)
    # calculate dim and shift
    if img.shape[0] > img.shape[1]:
        ratio = float(IMAGE_SIZE) / img.shape[0]
        dim = (int(img.shape[1] * ratio), IMAGE_SIZE)
        shift = np.float32([[1, 0, (dim[1] - dim[0]) / 2], [0, 1, 0]])
    else:
        ratio = float(IMAGE_SIZE) / img.shape[1]
        dim = (IMAGE_SIZE, int(img.shape[0] * ratio))
        shift = np.float32([[1, 0, 0], [0, 1, (dim[0] - dim[1]) / 2]])
    # resize the image to (IMAGE_SIZE, IMAGE_SIZE)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
    # zero padding the border
    img = cv2.copyMakeBorder(img, 0, IMAGE_SIZE - img.shape[0], 0, IMAGE_SIZE - img.shape[1], cv2.BORDER_CONSTANT,
                             value=255)
    # shift the image for centering
    img = cv2.warpAffine(img, shift, (IMAGE_SIZE, IMAGE_SIZE), borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    #plt.imshow(img, cmap='gray')
    #plt.show()
    img = cv2.blur(img, (5, 5))
    ret, img = cv2.threshold(img, 160, 255, cv2.THRESH_TRUNC)
    img = cv2.blur(img, (5, 5))
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #plt.imshow(img, cmap='gray')
    #plt.show()
    #img = cv2.fastNlMeansDenoising(img, None, 10, 21, 7)
    kernel1 = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel1)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img = cv2.erode(img, kernel2, iterations=3)
    #plt.imshow(img, cmap='gray')
    #plt.show()
    img = cv2.resize(img, (IMAGE_SIZE_s, IMAGE_SIZE_s), interpolation=cv2.INTER_LANCZOS4)
    for i in range(2):
        img = cv2.blur(img, (3, 3))
    #plt.imshow(img, cmap='gray')
    #plt.show()
    return img


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def buildNetwork(train_x, train_y, validate_x, validate_y, test_x):

    # placeholders
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE_s, IMAGE_SIZE_s, 1])
    y_ = tf.placeholder(tf.float32, shape=[None, 2])

    # first layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    # x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # third layer
    W_conv3 = weight_variable([5, 5, 64, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    # densely connected layer
    W_fc1 = weight_variable([32 * 32 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool3_flat = tf.reshape(h_pool3, [-1, 32 * 32 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # readout layer
    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])
    y_pred = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # loss function
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_pred))
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #saver.restore(sess, MODEL_PATH)
        #print("Model restored from file: %s" % MODEL_PATH)

        for e in range(1, EPOCH+1):
            avg_cost = 0.0
            for i in range(0, TRAIN_NUM-VALID_NUM, BATCH_SIZE):
                x_batch = train_x[i: i + BATCH_SIZE, :]
                y_batch = train_y[i: i + BATCH_SIZE, :]
                # do randomly data augmentation
                x_batch = dataAugmentation(x_batch)
                _, cost = sess.run([train_step, loss], feed_dict={x: x_batch, y_: y_batch, keep_prob: 0.5})
                avg_cost += cost*BATCH_SIZE/(TRAIN_NUM-VALID_NUM)
                if i % 1000 == 0:
                    validate_accuracy = accuracy.eval(feed_dict={x: validate_x, y_: validate_y, keep_prob: 1.0})
                    print("epoch {0} | step {1} | validation accuracy: {2}" .format(e, i, validate_accuracy))
            train_accuracy = accuracy.eval(feed_dict={x: train_x[:TRAIN_NUM], y_: train_y[:TRAIN_NUM], keep_prob: 1.0})
            print("epoch {0} | training loss {1} | training accuracy {2}".format(e, avg_cost, train_accuracy))
            save_path = saver.save(sess, MODEL_PATH)
            print("Model saved in file: %s" % save_path)


def dataAugmentation(train_x):
    angle = 180
    shift = 0.2
    zoom = 0.2
    dataGen = ImageDataGenerator(rotation_range=angle, width_shift_range=shift, height_shift_range=shift,
                                 zoom_range=zoom, fill_mode='constant', cval=255)
    dataGen.fit(train_x)
    return train_x


def saveImage(train_x, train_name):
    for i in range(TRAIN_NUM):
        path = train_name[i].split('/')
        side = path[3] + '_process_morphology'
        name = path[4].split('.')[0] + '_p.png'
        path_new = './1/train/'+side+'/'+name
        cv2.imwrite(path_new, train_x[i])


def main():
    # prepare data
    train_x, test_x, train_name = readData()
    #saveImage(train_x, train_name)
    train_y_left = np.array(([1] * (TRAIN_NUM/2)) + ([0] * (TRAIN_NUM/2)))
    train_y_right = np.array(([0] * (TRAIN_NUM/2)) + ([1] * (TRAIN_NUM/2)))
    train_y = np.transpose(np.vstack((train_y_left, train_y_right)))
    shuffled_x, shuffled_y = randomize(train_x, train_y)
    validate_x = shuffled_x[-VALID_NUM:]
    validate_y = shuffled_y[-VALID_NUM:]

    '''
    # sample display
    for i in range(TRAIN_NUM):
        img = train_x[i, :, :, 0]
        plt.imshow(img, cmap='gray')
        plt.show()
    '''
    buildNetwork(shuffled_x, shuffled_y, validate_x, validate_y, test_x)


if __name__ == '__main__':
    main()