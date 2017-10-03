#coding=utf-8

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.visualize_util import plot


def run():
    # 构建神经网络
    model = Sequential()
    model.add(Dense(4, input_dim=2, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dense(2, init='uniform'))
    model.add(Activation('sigmoid'))
    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # 神经网络可视化
    plot(model, to_file='model.png')


if __name__ == '__main__':
    run()

    # if __name__=='__main__':
#     import argparse
#     parser = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('--data-npz', dest='data_npz',
#                         help='.npz file including instances of DataSetCSVslotTagging for train, dev and test')
#     parser.add_argument('--loss', dest='loss',
#                         default='categorical_crossentropy',
#                         help='objective function')
#     parser.add_argument('--optimizer', dest='optimizer',
#                         default='adam', help='optimizer')
#     parser.add_argument('--epoch-nb', dest='epoch_nb', type=int,
#                         default=300, help='number of epoches')
#     parser.add_argument('--embedding-size', dest='embedding_size', type=int,
#                         default=512, help='the dimention of word embeddings.')
#     parser.add_argument('--patience', dest='patience', type=int,
#                         default=10, help='the patience for early stopping criteria')
#     parser.add_argument('--batch-size', dest='batch_size', type=int,
#                         default=32, help='batch size')
#     parser.add_argument('--hidden-size', dest='hidden_size', type=int,
#                         default=128, help='the number of hidden units in recurrent layer')
#     parser.add_argument('--dropout-ratio', dest='dropout_ratio',
#                         type=float, default=0.5, help='dropout ratio')
#     parser.add_argument('--model-folder', dest='model_folder',
#                         help='the folder contains graph.yaml, weights.h5, and other_vars.npz')
#     parser.add_argument('--test-tag', dest='test_tag_only', action='store_true',
#                         help='only perform user Tagging test if this option is activated.')
#     parser.add_argument('--test-intent', dest='test_intent_only', action='store_true',
#                         help='only perform user intent test if this option is activated.')
#     parser.add_argument('--train', dest='train_only', action='store_true',
#                         help='only perform training if this option is activated.')
#     parser.add_argument('--weights-file', dest='weights_fname', help='.h5 weights file.')
#     parser.add_argument('--threshold', dest='threshold', type=float, help='float number of threshold for multi-label prediction decision.')
#     args = parser.parse_args()
#     print(args)