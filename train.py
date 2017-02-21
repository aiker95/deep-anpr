#!/usr/bin/env python
#
# Copyright (c) 2016 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.


"""
Routines for training the network.

"""


__all__ = (
    'train',
)


import functools
import glob
from os import walk
import itertools
import multiprocessing
import random
import sys
import time

import cv2
import numpy
import tensorflow as tf

import common
#import gen
import model


def code_to_vec(p, code):
    def char_to_vec(c):
        y = numpy.zeros((len(common.CHARS),))
        y[common.CHARS.index(c)] = 1.0
        return y

    c = numpy.vstack([char_to_vec(c) for c in code])

    return numpy.concatenate([[1. if p else 0], c.flatten()])

def read_data():
    while True:
        # if random.randint(0, 100)<10 :
        fname = sys.stdin.readline().strip()
        #print "=", fname, "|"
        try:
            im = cv2.imread(fname)
            im = cv2.copyMakeBorder(im, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[255.0]) 
            im = cv2.resize(im, (128, 64))[:, :, 0].astype(numpy.float32) / 255.
        except:
            print "fail on ", fname
            continue
        if "/new" in fname:
            code = fname.split("/")[-1][9:18]
        else:
            code = fname.split("/")[-2]
        if len(code)==8:
            code += "_"
        p = 1
        if "/bad/" in fname:
            p = 0
            code = "A000AA000"
        # else:
        #     import gen
        #     p = False
        #     while not p :
        #         im, code, p = gen.generate_im(100000)
        #     p = 1
        #     if len(code)==8:
        #         code = code +"_"

        #print code , " ", p, " "
        yield im, code_to_vec(p, code)


def read_validation_data():
    f = []
    for (dirpath, dirnames, filenames) in walk("validation"):
        f.extend(filenames)
        break
    for fname in f:
        #print "=", fname, "|"
        im = cv2.imread("validation/"+fname)
        im = cv2.copyMakeBorder(im, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[255.0])
        im = cv2.resize(im, (128, 64))[:, :, 0].astype(numpy.float32) / 255.

        tmp = fname.split("/")[-1].split("-")
        code = tmp[1]
        if len(code)==8:
            code += "_"
        p = len(code)==9
        print code , " ", p
        yield im, code_to_vec(p, code)


def unzip(b):
    xs, ys = zip(*b)
    xs = numpy.array(xs)
    ys = numpy.array(ys)
    return xs, ys



def read_batches(batch_size):
    while True:
        b = list(itertools.islice(read_data(), batch_size))
        yield unzip(b)


def train(learn_rate, report_steps, batch_size, initial_weights=None):
    """
    Train the network.

    The function operates interactively: Progress is reported on stdout, and
    training ceases upon `KeyboardInterrupt` at which point the learned weights
    are saved to `weights.npz`, and also returned.

    :param learn_rate:
        Learning rate to use.

    :param report_steps:
        Every `report_steps` batches a progress report is printed.

    :param batch_size:
        The size of the batches used for training.

    :param initial_weights:
        (Optional.) Weights to initialize the network with.

    :return:
        The learned network weights.

    """
    x, y, params = model.get_training_model()

    y_ = tf.placeholder(tf.float32, [None, 9 * len(common.CHARS) + 1])
    keep_prob = tf.placeholder(tf.float32)
    digits_loss = tf.nn.softmax_cross_entropy_with_logits(
                                          tf.reshape(y[:, 1:],
                                                     [-1, len(common.CHARS)]),
                                          tf.reshape(y_[:, 1:],
                                                     [-1, len(common.CHARS)]))
    digits_loss = tf.reduce_sum(digits_loss)
    presence_loss = 10. * tf.nn.sigmoid_cross_entropy_with_logits(
                                                          y[:, :1], y_[:, :1])
    presence_loss = tf.reduce_sum(presence_loss)
    cross_entropy = digits_loss + presence_loss
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy)

    best = tf.argmax(tf.reshape(y[:, 1:], [-1, 9, len(common.CHARS)]), 2)
    correct = tf.argmax(tf.reshape(y_[:, 1:], [-1, 9, len(common.CHARS)]), 2)

    if initial_weights is not None:
        assert len(params) == len(initial_weights)
        assign_ops = [w.assign(v) for w, v in zip(params, initial_weights)]

    init = tf.initialize_all_variables()

    def vec_to_plate(v):
        return "".join(common.CHARS[i] for i in v)

    def do_report():
        r = sess.run([best,
                      correct,
                      tf.greater(y[:, 0], 0),
                      y_[:, 0],
                      digits_loss,
                      presence_loss,
                      cross_entropy],
                     feed_dict={x: test_xs, y_: test_ys, keep_prob: 1.0})
        num_correct = numpy.sum(
                        numpy.logical_or(
                            numpy.all(r[0] == r[1], axis=1),
                            numpy.logical_and(r[2] < 0.5,
                                              r[3] < 0.5)))
        r_short = (r[0][:190], r[1][:190], r[2][:190], r[3][:190])
        for b, c, pb, pc in zip(*r_short):
            print "{} {} <-> {} {}".format(vec_to_plate(c), pc,
                                           vec_to_plate(b), float(pb))
        num_p_correct = numpy.sum(r[2] == r[3])
        good = 100. * num_correct / (len(r[0]))
        print ("B{:3d} {:2.02f}% {:02.02f}% loss: {} "
               "(digits: {}, presence: {}) |{}|").format(
            batch_idx,
            good,
            100. * num_p_correct / len(r[2]),
            r[6],
            r[4],
            r[5],
            "".join("X "[numpy.array_equal(b, c) or (not pb and not pc)]
                                           for b, c, pb, pc in zip(*r_short)))
        return r[6]

    #def do_report():
    #    feed_dict = {x: batch_xs, y_: batch_ys, keep_prob: 0.5}
    #    return do_report()

    def do_batch():
        feed_dict = {x: batch_xs, y_: batch_ys, keep_prob: 0.5}
        sess.run(train_step, feed_dict)
        rs = 1000
        if batch_idx % report_steps == 0:
            rs = do_report()
        return rs

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)
        if initial_weights is not None:
            sess.run(assign_ops)


        batch_idx = 0
        test_xs, test_ys = unzip(read_validation_data())
        print "after read_data"
        do_report()
        best_loss = 240
        try:
            last_batch_idx = 0
            last_batch_time = time.time()
            print "before read batch"
            batch_iter = read_batches(batch_size)
            print "after read batch"
            for (batch_xs, batch_ys) in batch_iter:

                batch_idx += 1
                loss = do_batch()
                if batch_idx>100 and loss<best_loss:
                    print "save and continue: best = "+ str(loss)
                    last_weights = [p.eval() for p in params]
                    best_loss = loss
                    numpy.savez("weights_best_"+str(loss)+".npz", *last_weights)
                if batch_idx==50000:
                    print "save and finish"
                    last_weights = [p.eval() for p in params]
                    numpy.savez("weights.npz", *last_weights)
                    break

                if batch_idx % report_steps == 0:
                    batch_time = time.time()
                    if last_batch_idx != batch_idx:
                        print "time for 60 batches {}".format(
                            60 * (last_batch_time - batch_time) /
                                            (last_batch_idx - batch_idx))
                        last_batch_idx = batch_idx
                        last_batch_time = batch_time

        except KeyboardInterrupt:
            last_weights = [p.eval() for p in params]
            numpy.savez("weights.npz", *last_weights)
            return last_weights


if __name__ == "__main__":
    if len(sys.argv) > 1:
        f = numpy.load(sys.argv[1])
        initial_weights = [f[n] for n in sorted(f.files,
                                                key=lambda s: int(s[4:]))]
    else:
        initial_weights = None

    train(learn_rate=0.001,
          report_steps=20,
          batch_size=50,
          initial_weights=initial_weights)

