# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import cv2
import random


class ChessBoardClassifier:
    def __init__(self):
        self.sess = tf.InteractiveSession()
        self.grid_width = 40
        self.ch = 4
        self.label_dim = 13  # rnbqkp # RNBQKP
        self.label_value_dict = {'r': 0, 'n': 1, 'b': 2, 'q': 3, 'k': 4, 'p': 5, '#': 6,
                                'R': 7, 'N': 8, 'B': 9, 'Q': 10, 'K': 11, 'P': 12}
        self.model_path = './model/chessboard_classifier_model'
        self.batch_size = 32
        self.epoch = 1000
        self.learning_rate = 0.01

        self.grid_input = tf.placeholder('float', [None, self.grid_width, self.grid_width, self.ch])
        self.label_input = tf.placeholder('float', [None, self.label_dim])
        self.grid_label = self.build_net(self.grid_input)
        self.loss = self.get_loss()
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())
        # self.saver = tf.train.Saver()

        # self.sess.run(tf.global_variables_initializer())

        # self.save_model()
        # self.restore_model()

    def get_loss(self):
        return tf.reduce_mean(tf.squared_difference(self.grid_label, self.label_input))

    def save_model(self):
        print("Model saved in : ", self.saver.save(self.sess, self.model_path))

    def restore_model(self):
        self.saver.restore(self.sess, self.model_path)
        print("Model restored.")

    def build_net(self, img):
        with tf.variable_scope('dense1'):
            dense1 = tf.layers.dense(tf.contrib.layers.flatten(img), activation=None, units=50)
        with tf.variable_scope('dense2'):
            dense2 = tf.layers.dense(dense1, activation=None, units=50)
        with tf.variable_scope('dense3'):
            dense3 = tf.layers.dense(dense2, activation=tf.nn.relu, units=self.label_dim)
        with tf.variable_scope('softmax'):
            softmax = tf.nn.softmax(dense3)
        return softmax

    def train(self):
        # optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        grids = glob.glob("./grids/*.png")
        print len(grids)
        for ep in range(self.epoch):
            random_grids = random.sample(grids, self.batch_size)
            # print random_grids
            grid_batch = [np.array(Image.open(g))/255.0 for g in random_grids]
            # grid_batch = [mpimg.imread(g) for g in random_grids]
            # print grid_batch[0]
            label_batch = [self.label_value_dict[g.split('./grids/')[1][0]] for g in random_grids]
            label_batch = tf.one_hot(label_batch, self.label_dim).eval()
            # print label_batch

            print self.sess.run(self.loss, feed_dict={self.grid_input: grid_batch,
                                                 self.label_input: label_batch})

            correct_prediction = tf.equal(tf.argmax(self.grid_label, 1), tf.argmax(label_batch, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            # print 'real:', tf.argmax(label_batch, 1).eval()
            # print 'pred:', tf.argmax(self.grid_label, 1).eval(feed_dict={self.grid_input: grid_batch,
            #                                self.label_input: label_batch})
            # print 'pred:', self.grid_label.eval(feed_dict={self.grid_input: grid_batch,
            #                                                              self.label_input: label_batch})
            print 'accuracy:', accuracy.eval(feed_dict={self.grid_input: grid_batch,
                                           self.label_input: label_batch})

            self.sess.run(self.optimize, feed_dict={self.grid_input: grid_batch,
                                                               self.label_input: label_batch})


    def get_grid_label(self, path):
        lines = path.split('/')[-1].split('.gif')[0].split('_')
        print lines
        labels = []
        for line in lines:
            for c in line:
                if c.isdigit():
                    labels.extend(['#' for _ in range(int(c))])
                else:
                    labels.append(c)

        return labels



    def get_grid_data(self):
        filenames = sorted(glob.glob("./chessGif/*.gif"))
        count = 0
        list = np.zeros(13)
        for filename in filenames:
            labels = self.get_grid_label(filename)
            board = mpimg.imread(filename)[4:324, 4:324, :]
            for i in range(8):
                for j in range(8):
                    grid = board[i*self.grid_width:i*self.grid_width+40, j*self.grid_width:j*self.grid_width+40, :]
                    # plt.imshow(grid)  # 显示图片
                    # plt.axis('off')  # 不显示坐标轴
                    # plt.show()
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
    classifier.train()
