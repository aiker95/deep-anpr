import sys
import signal
from requests import ConnectionError
from threading import Thread
from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
import cv2
import numpy
import Queue
import threading
import tensorflow as tf
import time

import common
import model
import collections

__all__ = (
    'detect',
    'post_process',
)


class POSTHandler(BaseHTTPRequestHandler):
    # Load the model which detects number plates over a sliding window.
    x, y, params = model.get_detect_model()
    config = tf.ConfigProto()
    sess = tf.Session()
    @staticmethod
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

        # Execute the model at each scale.
        feed_dict = {POSTHandler.x: numpy.stack([im])}
        feed_dict.update(dict(zip(POSTHandler.params, param_vals)))

        y_val = POSTHandler.sess.run(POSTHandler.y, feed_dict=feed_dict)

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

        letter_probs_softmax = common.softmax(letter_probs)

        # img_scale = float(im.shape[0]) / im.shape[0]

        # bbox_tl = window_coords * (8, 4) * img_scale
        # bbox_size = numpy.array(model.WINDOW_SHAPE) * img_scale

        present_prob = common.sigmoid(
            y_val[0, window_coords[0], window_coords[1], 0])

        # yield bbox_tl, bbox_tl + bbox_size, present_prob, letter_probs
        return present_prob, letter_probs_softmax

    @staticmethod
    def _overlaps(match1, match2):
        bbox_tl1, bbox_br1, _, _ = match1
        bbox_tl2, bbox_br2, _, _ = match2
        return (bbox_br1[0] > bbox_tl2[0] and
                bbox_br2[0] > bbox_tl1[0] and
                bbox_br1[1] > bbox_tl2[1] and
                bbox_br2[1] > bbox_tl1[1])

    @staticmethod
    def _group_overlapping_rectangles(matches):
        matches = list(matches)
        num_groups = 0
        match_to_group = {}
        for idx1 in range(len(matches)):
            for idx2 in range(idx1):
                if POSTHandler._overlaps(matches[idx1], matches[idx2]):
                    match_to_group[idx1] = match_to_group[idx2]
                    break
            else:
                match_to_group[idx1] = num_groups
                num_groups += 1

        groups = collections.defaultdict(list)
        for idx, group in match_to_group.items():
            groups[group].append(matches[idx])

        return groups

    @staticmethod
    def letter_probs_to_code(letter_probs):
        return "".join(common.CHARS[i] for i in numpy.argmax(letter_probs, axis=1))

    q = Queue.Queue()

    @staticmethod
    def worker():
        f = numpy.load("weights.npz")
        param_vals = [f[n] for n in sorted(f.files, key=lambda s: int(s[4:]))]
        lastPlates = dict()
        while True:
            item, camId, fname = POSTHandler.q.get()
            try:
                import time
                im_gray = cv2.resize(item, (128, 64))
                start = time.time()
                present_prob, letter_probs = POSTHandler.detect(im_gray, param_vals)
                end = time.time()
                code = POSTHandler.letter_probs_to_code(letter_probs)
                oldTime, oldCode = lastPlates.get(camId, (0, ""))
                now = time.time()
                import datetime
                strNow = datetime.datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S.%f')
                if code == oldCode and now - oldTime<30:
                    print strNow,  " skip repeated plate: ", code, "; file: ", fname, "; time: ", now-oldTime, "; CNN time: ", (end-start)
                else:
                    print strNow, " post license plate: ", code, "; file: ", fname, "; CNN time: ", (end-start)
                    lastPlates[camId] = (now, code)
                    import requests
                    headers = {'Content-type': 'application/json'}
                    #r = requests.post("http://localhost/api/service/platerecognition", data={'number': code, 'id': camId})
                    r = requests.post("https://green-pay.net/api/service/platerecognition", data={'number': code, 'id': camId}, headers=headers)
                    print present_prob, " ", code, " ", r.status_code, r.reason
                    sys.stdout.flush()
            except ConnectionError as e:
                print "Network problem; ", e
                lastPlates[camId] = (0, "")
            except:
                import traceback
                traceback.print_exc()
                lastPlates[camId] = (0, "")

    @staticmethod
    def startWorkerThread():
        t = threading.Thread(target=POSTHandler.worker)
        t.daemon = True
        t.start()
        return t

    def do_POST(self):
        #print "----- SOMETHING WAS POST!! ------"
        #print self.headers
        length = int(self.headers['Content-Length'])
        anprPiNumber = self.headers['Anpr-Number']
        anprPiProbability = self.headers['Anpr-Probability']
        camId = self.headers['Camera-ID']
        imgTime = self.headers['Image-Time']
        content = self.rfile.read(length)
        print "Number: ", anprPiNumber, "; prop: ", anprPiProbability, "; Length: ", len(content)
        if len(content) != length:
            print "len1 != len2 : ", length, "!=", len(content)
            self.send_response(400)
            return
        fname = 'images/' + camId + "-" + str(imgTime) + "-" + anprPiNumber.replace("?",
                                                                                   "_") + "-" + anprPiProbability + ".png"
        f = open(fname, 'w')
        try:
            f.write(content)
        finally:
            f.close()
        image = cv2.imdecode(numpy.frombuffer(content, dtype='ubyte'), 0)
        triple = (image, camId, fname)
        POSTHandler.q.put(triple)


def run_on(port):
    print("Starting a server on port %i" % port)
    server_address = ('0.0.0.0', port)
    POSTHandler.startWorkerThread()
    httpd = HTTPServer(server_address, POSTHandler)
    httpd.serve_forever()


if __name__ == "__main__":
    ports = [int(arg) for arg in sys.argv[1:]]
    for port_number in ports:
        server = Thread(target=run_on, args=[port_number])
        server.daemon = True  # Do not make us wait for you to exit
        server.start()
    signal.pause()  # Wait for interrupt signal, e.g. KeyboardInterrupt
