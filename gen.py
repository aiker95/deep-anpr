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
Generate training and test images.

"""


__all__ = (
    'generate_ims',
)


import math
import os
import random
import sys

import cv2
import numpy

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

DIGITS  = "0123456789"
LETTERS = "ABCEHKMOPTXY"


FONT_PATH = "RoadNumbers2.0.ttf"
FONT_HEIGHT = 18  # Pixel size to which the chars are resized

OUTPUT_SHAPE = (50, 150)




def euler_to_mat(yaw, pitch, roll):
    # Rotate clockwise about the Y-axis
    c, s = math.cos(yaw), math.sin(yaw)
    M = numpy.matrix([[  c, 0.,  s],
                      [ 0., 1., 0.],
                      [ -s, 0.,  c]])

    # Rotate clockwise about the X-axis
    c, s = math.cos(pitch), math.sin(pitch)
    M = numpy.matrix([[ 1., 0., 0.],
                      [ 0.,  c, -s],
                      [ 0.,  s,  c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(roll), math.sin(roll)
    M = numpy.matrix([[  c, -s, 0.],
                      [  s,  c, 0.],
                      [ 0., 0., 1.]]) * M

    return M


def pick_colors():
    first = True
    while first or plate_color - text_color < 0.3:
        text_color = random.random()
        plate_color = random.random()
        if text_color > plate_color:
            text_color, plate_color = plate_color, text_color
        first = False
    return text_color, plate_color


def make_affine_transform(from_shape, to_shape,
                          min_scale, max_scale,
                          scale_variation=1.0,
                          rotation_variation=1.0,
                          translation_variation=1.0):
    out_of_bounds = False

    from_size = numpy.array([[from_shape[1], from_shape[0]]]).T
    to_size = numpy.array([[to_shape[1], to_shape[0]]]).T

    scale = random.uniform((min_scale + max_scale) * 0.5 -
                           (max_scale - min_scale) * 0.5 * scale_variation,
                           (min_scale + max_scale) * 0.5 +
                           (max_scale - min_scale) * 0.5 * scale_variation)
    if scale > max_scale*1.1 or scale < min_scale:
        #print "out #1 ", scale, max_scale, min_scale
        out_of_bounds = True
    roll = random.uniform(-0.15, 0.15) * rotation_variation
    pitch = random.uniform(-0.1, 0.1) * rotation_variation
    yaw = random.uniform(-1.1, 1.1) * rotation_variation

    # Compute a bounding box on the skewed input image (`from_shape`).
    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    h, w = from_shape
    corners = numpy.matrix([[-w, +w, -w, +w],
                            [-h, -h, +h, +h]]) * 0.5
    skewed_size = numpy.array(numpy.max(M * corners, axis=1) -
                              numpy.min(M * corners, axis=1))

    # Set the scale as large as possible such that the skewed and scaled shape
    # is less than or equal to the desired ratio in either dimension.
    scale *= numpy.min(to_size / skewed_size)

    # Set the translation such that the skewed and scaled image falls within
    # the output shape's bounds.
    trans = (numpy.random.random((2,1)) - 0.5) * translation_variation
    trans = ((2.0 * trans) ** 5.0) / 2.0
    if numpy.any(trans < -0.5) or numpy.any(trans > 0.5):
        out_of_bounds = True
    trans = (to_size - skewed_size * scale) * trans

    center_to = to_size / 2.
    center_from = from_size / 2.
    M = euler_to_mat(yaw, pitch, roll)[:2, :2]
    M *= scale
    M = numpy.hstack([M, trans + center_to - M * center_from])

    return M, out_of_bounds


def generate_code():
    if random.choice((True, False)) :
        return "{}{}{}{}{}{}{}{}".format(
            random.choice(LETTERS),
            random.choice(DIGITS),
            random.choice(DIGITS),
            random.choice(DIGITS),
            random.choice(LETTERS),
            random.choice(LETTERS),
            random.choice(DIGITS),
            random.choice(DIGITS))
    else:
        return "{}{}{}{}{}{}{}".format(
            random.choice(LETTERS),
            random.choice(DIGITS),
            random.choice(DIGITS),
            random.choice(DIGITS),
            random.choice(LETTERS),
            random.choice(LETTERS),
            random.choice(["102", "103", "113", "116", "121", "123", "124", "125", "126", "134",
                           "136", "138", "142", "150", "190", "750", "152", "154", "159", "161",
                           "163", "164", "196", "173", "174", "177", "197", "199", "777", "178",
                           "186"]))



def rounded_rect(shape, radius):
    out = numpy.ones(shape)
    out[:radius, :radius] = 0.0
    out[-radius:, :radius] = 0.0
    out[:radius, -radius:] = 0.0
    out[-radius:, -radius:] = 0.0

    cv2.circle(out, (radius, radius), radius, 1.0, -1)
    cv2.circle(out, (radius, shape[0] - radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, shape[0] - radius), radius, 1.0, -1)

    return out

def set_numbers(svg, numbers='m976mm34'):
    import xml.etree.ElementTree as ET
    tree = ET.fromstring(svg)

    for elem in tree.iter('{http://www.w3.org/2000/svg}text'):
        id = elem.attrib['id']
        if id.startswith('plate'):
            text = ''
            for c in id[5:]:
                text += numbers[int(c)]
            elem[0].text = text
    return ET.tostring(tree)

def generate_plate():
    code = generate_code()
    import cairo
    import rsvg
    img = cairo.ImageSurface(cairo.FORMAT_ARGB32, 520, 112)
    ctx = cairo.Context(img)
    if len(code)==8 :
        codeSvg = open(os.path.join(os.path.dirname(__file__), 'ru.svg'), 'r').read()
    elif len(code)==9 :
        codeSvg = open(os.path.join(os.path.dirname(__file__), 'ru2.svg'), 'r').read()
    else:
        exit(-1)
    codeSvg= set_numbers(codeSvg, numbers=code)
    handle = rsvg.Handle(None, codeSvg)
    handle.render_cairo(ctx)

    buf = img.get_data()
    a = numpy.frombuffer(buf, numpy.uint8)
    a.shape = (112, 520, 4)
    a[:, :, 2] = 255
    a = cv2.resize(a, (150,50))
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) / 255.0
    out_shape = (50, 150)
    radius = 4
    return a, rounded_rect(out_shape, radius), code.replace(" ", "")


def generate_bg(num_bg_images):
    found = False
    while not found:
        fname = "bgs/{:08d}.jpg".format(random.randint(0, num_bg_images - 1))
        bg = cv2.imread(fname, 0) / 255.
        if (bg.shape[1] >= OUTPUT_SHAPE[1] and
            bg.shape[0] >= OUTPUT_SHAPE[0]):
            found = True

    x = random.randint(0, bg.shape[1] - OUTPUT_SHAPE[1])
    y = random.randint(0, bg.shape[0] - OUTPUT_SHAPE[0])
    bg = bg[y:y + OUTPUT_SHAPE[0], x:x + OUTPUT_SHAPE[1]]

    return bg


def generate_im(num_bg_images):
    bg = generate_bg(num_bg_images)

    plate, plate_mask, code = generate_plate()

    M, out_of_bounds = make_affine_transform(
                            from_shape=plate.shape,
                            to_shape=bg.shape,
                            min_scale=0.7,
                            max_scale=1,
                            rotation_variation=0.8,
                            scale_variation=1.5,
                            translation_variation=1.2)
    plate = cv2.warpAffine(plate, M, (bg.shape[1], bg.shape[0]))
    #plate_mask = cv2.warpAffine(plate_mask, M, (bg.shape[1], bg.shape[0]))

    out = plate*0.6 + bg*0.4
    #out = plate * plate_mask + bg * (1.0 - plate_mask)/2 + bg/2

    out = cv2.resize(out, (OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]))

    out += numpy.random.normal(scale=0.05, size=out.shape)
    out = numpy.clip(out, 0., 1.)

    return out, code, not out_of_bounds


def generate_ims(num_images):
    """
    Generate a number of number plate images.

    :param num_images:
        Number of images to generate.

    :return:
        Iterable of number plate images.

    """
    variation = 1.0
    num_bg_images = len(os.listdir("bgs"))
    for i in range(num_images):
        yield generate_im(num_bg_images)


if __name__ == "__main__":
    os.mkdir("test")
    im_gen = generate_ims(int(sys.argv[1]))
    for img_idx, (im, c, p) in enumerate(im_gen):
        fname = "test/{}/{:08d}_{}_{}.png".format(c, img_idx, c,
                                               "1" if p else "0")
        print (fname)
        if p:
            if not os.path.exists("test/"+c) : os.mkdir("test/"+c)
            cv2.imwrite(fname, im * 255.)

