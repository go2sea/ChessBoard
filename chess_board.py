# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random


class ChessBoardClassifier:
    def __init__(self):
        self.sess = tf.InteractiveSession()
        self.ch = 4
        self.label_dim = 13  # rnbqkp # RNBQKP
        self.label_value_dict = {'r': 0, 'n': 1, 'b': 2, 'q': 3, 'k': 4, 'p': 5, '#': 6,
                                'R': 7, 'N': 8, 'B': 9, 'Q': 10, 'K': 11, 'P': 12}
        self.model_path = './model/chessboard_classifier_model'
        self.batch_size = 64
        self.epoch = 1000
        self.learning_rate = 0.001

        self.grid_input = tf.placeholder('float', [None, 40, 40, self.ch])
        self.label_input = tf.placeholder('float', [None, self.label_dim])
        self.grid_label = self.build_net(self.grid_input)
        self.loss = tf.reduce_mean(tf.squared_difference(self.grid_label, self.label_input))
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        self.restore_model()

    def save_model(self):
        print("Model saved in : ", self.saver.save(self.sess, self.model_path))

    def restore_model(self):
        self.saver.restore(self.sess, self.model_path)
        print("Model restored.")

    def build_net(self, img):
        with tf.variable_scope('conv1'):
            relu1 = tf.layers.conv2d(img, filters=6, kernel_size=[3, 3], strides=[2, 2], padding='SAME', activation=tf.nn.relu)
        with tf.variable_scope('conv2'):
            relu2 = tf.layers.conv2d(relu1, filters=12, kernel_size=[3, 3], strides=[2, 2], padding='SAME', activation=tf.nn.relu)
        with tf.variable_scope('dense1'):
            dense1 = tf.layers.dense(tf.contrib.layers.flatten(relu2), activation=None, units=50)
        with tf.variable_scope('dense2'):
            dense2 = tf.layers.dense(dense1, activation=None, units=50)
        with tf.variable_scope('dense3'):
            dense3 = tf.layers.dense(dense2, activation=None, units=self.label_dim)
        with tf.variable_scope('softmax'):
            return tf.nn.softmax(dense3)

    def train(self):
        grids = glob.glob("./grids/*.png")
        random.shuffle(grids)
        for ep in range(self.epoch):
            random_grids = random.sample(grids, self.batch_size)
            grid_batch = [np.array(Image.open(g))/255.0 + np.random.normal(0.0, 0.05, np.array(Image.open(g)).shape) for g in random_grids]
            label_batch = [self.label_value_dict[g.split('./grids/')[1][0]] for g in random_grids]
            label_batch = tf.one_hot(label_batch, self.label_dim).eval()
            print self.sess.run(self.loss, feed_dict={self.grid_input: grid_batch, self.label_input: label_batch})
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.grid_label, 1), tf.argmax(label_batch, 1)), tf.float32))
            print 'accuracy:', accuracy.eval(feed_dict={self.grid_input: grid_batch, self.label_input: label_batch})
            self.sess.run(self.optimize, feed_dict={self.grid_input: grid_batch, self.label_input: label_batch})
            if ep % 100 == 0:
                self.save_model()

    def predict(self, path):
        wrong_prediction = 0
        total_prediction = 0
        boards = glob.glob("{}/*.gif".format(path))
        for board_path in boards:
            board = mpimg.imread(board_path)[4:324, 4:324, :]/255.0
            labels = self.get_grid_label(board_path)
            for i in range(8):
                for j in range(8):
                    total_prediction += 1
                    grid = board[i*40:(i+1)*40, j*40:(j+1)*40, :]
                    label = self.sess.run(self.grid_label, feed_dict={self.grid_input: [grid]})[0]
                    wrong_prediction += np.argmax(label) != self.label_value_dict[labels[i * 8 + j]]
        print 'wrong_prediction: {}/{}'.format(wrong_prediction, total_prediction)

    def get_grid_label(self, path):
        lines = path.split('/')[-1].split('.gif')[0].split('_')
        labels = []
        for line in lines:
            for c in line:
                labels.extend(['#' for _ in range(int(c))] if c.isdigit() else [c])
        return labels

    def get_grid_data(self):
        filenames = sorted(glob.glob("./chessGif/*.gif"))
        count = 0
        list = np.zeros(self.label_dim)
        for filename in filenames:
            labels = self.get_grid_label(filename)
            board = mpimg.imread(filename)[4:324, 4:324, :]
            for i in range(8):
                for j in range(8):
                    grid = board[i*40:i*40+40, j*40:j*40+40, :]
                    list[self.label_value_dict[labels[i*8 + j]]] += 1
                    if list[self.label_value_dict[labels[i*8 + j]]] <= 200:
                        plt.imsave('./grids/{}_{}.png'.format(labels[i*8 + j], count), grid)
                    count += 1
        # print board.shape
        # plt.imshow(board)  # 显示图片
        # plt.axis('off')  # 不显示坐标轴
        # plt.show()


if __name__ == '__main__':
    classifier = ChessBoardClassifier()
    # classifier.get_grid_data()
    # classifier.train()
    classifier.predict('./val_data')
