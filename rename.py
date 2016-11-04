#!/usr/bin/env python

import os

bg_images = os.listdir("bgs")
i = 0
for img in bg_images:
   os.rename("bgs/"+img, "bgs/{:08d}.jpg".format(i))
   i += 1