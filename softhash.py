#!/usr/bin/python

import cv2
import numpy
from matplotlib import pyplot, cm
import random

import matplotlib
matplotlib.interactive(True)
matplotlib.use("TkAgg")
# pyplot.ion()

DEBUG = True


class SoftHash(object):

    _block_size = 8     # use an 8x8 blocksize for the DCT
    _img = None    # the original image we are working on
    _img_cropped = None  # the cropped image since we work on 8x8 blocks
    _color_image = False  # tells if the image is grayscale or color

    """ _key is a parameter which tells us how many blocks we select
        for ease of use we generate the blocks with the mersenne twister
        python RNG based on a small key that can be initialized"""
    _key = []

    _keysize = 8  # the number of selected blocks + some other fun stuff

    _blocks = 0  # the number of blocks (of _block_size) composing the image

    def __init__(self, imagefile, key, blocksize=8):

        self._block_size = blocksize
        self._img = cv2.imread(imagefile, cv2.CV_LOAD_IMAGE_UNCHANGED)
        # cv2 loads images in BGR. BGR -> RGB conversion needed

        if self._img.shape[2] == 3:
            self._color_image = True
            # it's a color image let's convert! :)
            self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2RGB)

        self._blocks = (numpy.array(self._img.shape[:2]) /
                        self._block_size)
        h, w = self._blocks * self._block_size
        # since we are using 256x256 pixel images this step
        # shouldn't be needed but it is here just for completeness
        print self._blocks
        print h, w

        self._img_cropped = self._img[:h, :w]

        self.initializeKey(key)

    def initializeKey(self, key):
        random.seed(key)  # initialize RNG

        blocks = self._blocks[0] * self._blocks[1]
        # get the total number of blocks and use it to get the random blocks
        self._key = random.sample(range(0, blocks), self._keysize)
        if DEBUG:
            print "Key is: %s" % self._key

    def plotImg(self, name=None):

        pyplot.imshow(self._img)

    def denoise(self):
        if self.is_color:
            imgfilter = cv2.fastNlMeansDenoisingColored
        else:
            imgfilter = cv2.fastNlMeansDenoising

        self.applyFilter(imgfilter)

    def applyFilter(self, imgfilter, *args, **kwargs):
        """
        apply the given filter to the image
        """
        imgfilter(self._img, self._img)
        pass

    def hash(self):
        # by default we denoise the image using non-local means algorithm
        self.denoise()

        # step 2 we can

    @property
    def is_color(self):
        return self._color_image


if __name__ == "__main__":
    sf = SoftHash('./ImageDatabaseCrops/NikonD60/DS-01-UTFI-0000-0_crop.TIF')
    # sf = SoftHash('Ub7XL8T.png', 1234)

    if DEBUG:
        pyplot.figure('original')
        sf.plotImg()

    sf.denoise()  # let's denoise it!

    # ok now let's

    if DEBUG:
        pyplot.figure('denoised')
        sf.plotImg()

    pyplot.show(block=True)


"""
looks like: 'The spatial filtering applied by the human visual system appears
to be low pass for chromatic stimuli and band pass for luminance stimuli'
http://www.ncbi.nlm.nih.gov/pubmed/9499586 so we decided to use the

so we are going to use the luminance and a RGB byte which is the DC component
of every channel to define the dominant color in the 8x8 block, this could help
discriminating images which have a similar luminance

hash structure:
- for each 8x8 block composing the picture we select the low frequencies
(using quantization -we could even use the JPEG quantization tables!-)
- the resulting hash size depends on:
    + # of 8x8 blocks used
    + ratio (we will resize the image given this parameter)
    + 1 int *R/G/B major component*

"""
