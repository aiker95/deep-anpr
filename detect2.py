#!/usr/bin/env python

"""
Routines to detect number plates.

"""


__all__ = (
    'detect',
    'post_process',
)


import collections
import itertools
import math
import sys

import cv2
import numpy
import tensorflow as tf

import common
import model



def detect(im, param_vals):
    """
    Detect number plates in an image.

    :param im:
        Image to detect number plates in.

    :param param_vals:
        Model parameters to use. These are the parameters output by the `train`
        module.

    :returns:
        Iterable of `bbox_tl, bbox_br, letter_probs`, defining the bounding box
        top-left and bottom-right corners respectively, and a 7,36 matrix
        giving the probability distributions of each letter.

    """

    # Load the model which detects number plates over a sliding window.
    x, y, params = model.get_detect_model()

    # Execute the model at each scale.
    with tf.Session(config=tf.ConfigProto()) as sess:
        feed_dict = {x: numpy.stack([im])}
        feed_dict.update(dict(zip(params, param_vals)))
        y_val = sess.run(y, feed_dict=feed_dict)

    # Interpret the results in terms of bounding boxes in the input image.
    # Do this by identifying windows (at all scales) where the model predicts a
    # number plate has a greater than 50% probability of appearing.
    #
    # To obtain pixel coordinates, the window coordinates are scaled according
    # to the stride size, and pixel coordinates.
        window_coords = numpy.array([0, 0, 64, 128])
        letter_probs = (y_val[0,
                              window_coords[0],
                              window_coords[1], 1:].reshape(
                                9, len(common.CHARS)))
        letter_probs = common.softmax(letter_probs)

        #img_scale = float(im.shape[0]) / im.shape[0]

        #bbox_tl = window_coords * (8, 4) * img_scale
        #bbox_size = numpy.array(model.WINDOW_SHAPE) * img_scale

        present_prob = common.sigmoid(
                           y_val[0, window_coords[0], window_coords[1], 0])

        #yield bbox_tl, bbox_tl + bbox_size, present_prob, letter_probs
        yield present_prob, letter_probs


def _overlaps(match1, match2):
    bbox_tl1, bbox_br1, _, _ = match1
    bbox_tl2, bbox_br2, _, _ = match2
    return (bbox_br1[0] > bbox_tl2[0] and
            bbox_br2[0] > bbox_tl1[0] and
            bbox_br1[1] > bbox_tl2[1] and
            bbox_br2[1] > bbox_tl1[1])


def _group_overlapping_rectangles(matches):
    matches = list(matches)
    num_groups = 0
    match_to_group = {}
    for idx1 in range(len(matches)):
        for idx2 in range(idx1):
            if _overlaps(matches[idx1], matches[idx2]):
                match_to_group[idx1] = match_to_group[idx2]
                break
        else:
            match_to_group[idx1] = num_groups 
            num_groups += 1

    groups = collections.defaultdict(list)
    for idx, group in match_to_group.items():
        groups[group].append(matches[idx])

    return groups

def letter_probs_to_code(letter_probs):
    return "".join(common.CHARS[i] for i in numpy.argmax(letter_probs, axis=1))


if __name__ == "__main__":

    f = numpy.load(sys.argv[1])
    param_vals = [f[n] for n in sorted(f.files, key=lambda s: int(s[4:]))]
    import sys
    while True:
        im = cv2.resize(cv2.imread(sys.stdin.readline().strip()), (128, 64))
        im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) / 255.
        for present_prob, letter_probs in detect(im_gray, param_vals):
            code = letter_probs_to_code(letter_probs)
            print present_prob, " ", code
            sys.stdout.flush()
