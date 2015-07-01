#!/usr/bin/python

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import random

import logging

import matplotlib
matplotlib.interactive(True)
matplotlib.use("TkAgg")
# plt.ion()

DEBUG = True

logger = logging.getLogger("softhash")
ch = logging.StreamHandler()

logger.setLevel("DEBUG")
ch.setLevel("DEBUG")

# logger.setLevel("WARNING")
# ch.setLevel("WARNING")

logger.addHandler(ch)


class SoftHash(object):

    """
        this class contains the implementation for the softhash,
        it works by initializing it giving a key and a blocksize,
        then one can always modify the "img" property and use the
        update method to get the new hash
        this way this can be used as a module and filters and other
        operations on the image can be added without modifying it
    """

    _block_size = 8     # use an 8x8 blocksize for the DCT
    _img = None    # the original image we are working on
    _hash = None    # contains the current hash

    @property
    def img(self):
        return self._img

    @property
    def key_pixels(self):
        return np.array(self._key) * self._block_size

    @img.setter
    def img(self, value):
        self._img = value

    _color_image = False  # tells if the image is grayscale or color

    """ _key is a parameter which tells us how many blocks we select
        for ease of use we generate the blocks with the mersenne twister
        python RNG based on a small key that can be initialized"""
    _key = []

    _keysize = 8  # the number of selected blocks

    # the number of blocks (of _block_size)
    # composing the image w/h
    _blocks = 0

    @property
    def _nblocks(self):
        """ the total number of blocks composing the image """
        return self._blocks[0] * self._blocks[1]

    def __init__(self, imagefile, key, blocksize=8, selectedblocks=8):

        self._block_size = blocksize
        self.img = cv2.imread(imagefile, cv2.CV_LOAD_IMAGE_UNCHANGED)
        # cv2 loads images in BGR. BGR -> RGB conversion needed

        if self.img.shape[2] == 3:
            self._color_image = True
            # it's a color image let's convert!
            # this way we can show it using plt

            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        self._blocks = (np.array(self.img.shape[:2]) /
                        self._block_size)

        h, w = self._blocks * self._block_size
        # since we are using 256x256 pixel images this step
        # shouldn't be needed but it is here just for completeness
        logger.debug("cropped image size")
        logger.debug("h, w: %s", (h, w))
        logger.debug("blocks: %s", self._blocks)

        self.img = self.img[:h, :w]

        self.initializeKey(key)

        if selectedblocks > self._nblocks:
            # ensure that we select at most the
            # number of blocks composing the image
            selectedblocks = self._nblocks

    def initializeKey(self, key):
        random.seed(key)  # initialize RNG

        # create the key selecting which blocks we are going to use
        print self._blocks
        self._key = [
            [random.choice(range(0, self._blocks[0])),
                random.choice(range(0, self._blocks[1]))]
            for i in range(0, self._keysize)
        ]

        logger.debug("Key is: %s", self._key)

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
        imgfilter(self.img, self.img)
        pass

    def update(self):
        # by default we denoise the image using non-local means algorithm
        # should denoising be done before or after resizing?
        self.denoise()

        # we will use the luminance channel
        # if it's greyscale just use the existing channel
        if self.is_color:
            _img = cv2.cvtColor(self.img, cv2.COLOR_RGB2YCR_CB)
            Y = np.zeros((_img.shape[0], _img.shape[1]), dtype=_img.dtype)
            Y[:, :] = _img[:, :, 0]

            # Cr = np.zeros((_img.shape[0], _img.shape[1]), dtype=_img.dtype)
            # Cr[:, :] = _img[:, :, 1]
            # Cb = np.zeros((_img.shape[0], _img.shape[1]), dtype=_img.dtype)
            # Cb[:, :] = _img[:, :, 2]

        else:
            Y = self.img

        logger.debug("Y: %s", Y)
        blocksize = self._block_size

        if DEBUG:
            plt.figure('Y channel')
            plt.imshow(Y, cmap=plt.get_cmap('gray'), interpolation=None)
            for block in self.key_pixels:
                bl = block[::-1]
                bl = [
                    [bl[0], bl[0], bl[0] + blocksize,
                        bl[0] + blocksize, bl[0]],
                    [bl[1], bl[1] + blocksize, bl[1] + blocksize, bl[1], bl[1]]
                ]

                plt.plot(bl[0], bl[1], 'r-')

        # plt.figure('Cr channel')
        # plt.imshow(Cr)
        # plt.figure('Cb channel')
        # plt.imshow(Cb)

        # now we use the selected block with the given key

        logger.debug("using blocks of size %d", self._block_size)

        if DEBUG:
            # needed to calc the square layout for plots
            sqr = np.around(len(self._key) / (2 * np.sqrt(2)))

        quant = np.rot90(np.triu(np.ones([blocksize, blocksize])))
        logger.debug("quantization matrix: %s", quant)

        for idx, block in enumerate(self.key_pixels):

            # now like the JPEG standard we shift the values by -128
            # don't know if it would change anything with cv2 dct
            # implementation... but looks like a good rule to follow
            # http://compgroups.net/comp.compression/level-shift-in-jpeg-optional-or-mandatory/175097

            B = np.zeros((blocksize, blocksize), dtype=np.int8)

            B[:, :] = Y[
                block[0]:block[0] + blocksize,
                block[1]:block[1] + blocksize] - 128

            logger.debug("index = %s", block)
            logger.debug("selected block: %s", B)

            Bdct = cv2.dct(np.array(B, dtype=np.float32))
            logger.debug("block dct: %s", Bdct)

            # quantize the block DCT cleaning out high frequencies
            Qdct = np.multiply(quant, Bdct)

            logger.debug("quantized block DCT: %s", Qdct)

            invBdct = cv2.idct(Qdct)

            if DEBUG:
                # show the selected blocks

                plt.figure('selected blocks')
                plt.subplot(sqr, sqr, idx + 1)
                plt.imshow(
                    B, cmap=plt.get_cmap('gray'),
                    interpolation='nearest')

                plt.figure('blocks DCT')
                plt.subplot(sqr, sqr, idx + 1)
                plt.imshow(
                    Bdct, cmap=plt.get_cmap('gray'),
                    interpolation='nearest')

                plt.figure('decoded DCT blocks')
                plt.subplot(sqr, sqr, idx + 1)
                plt.imshow(
                    invBdct, cmap=plt.get_cmap('gray'),
                    interpolation='nearest')

        if DEBUG:
            plt.tight_layout()

    @property
    def is_color(self):
        return self._color_image


if __name__ == "__main__":
    # sf = SoftHash(
    #    './ImageDatabaseCrops/NikonD60/DS-01-UTFI-0196-0_crop.TIF',
    #    1234)
    sf = SoftHash('test.png', 1234, 8, 32)

    sf.update()

    # TODO: based on the keysize we can decide to resize the image
    # to match the final key size :)

    plt.show(block=True)


"""
'The spatial filtering applied by the human visual system appears
to be low pass for chromatic stimuli and band pass for luminance stimuli'
http://www.ncbi.nlm.nih.gov/pubmed/9499586

so we are going to use the luminance

hash structure:
- for each 8x8 block composing the picture we select the low frequencies
(using quantization -we could even use the JPEG quantization tables!-)
- the resulting hash size depends on:
    + # of 8x8 blocks used
    + ratio (we will resize the image given this parameter)

"""
