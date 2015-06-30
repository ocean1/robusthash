#!/usr/bin/python

import cv2
import numpy
from matplotlib import pyplot, cm
import random

import logging

import matplotlib
matplotlib.interactive(True)
matplotlib.use("TkAgg")
# pyplot.ion()

DEBUG = True

logger = logging.getLogger("softhash")
ch = logging.StreamHandler()
if DEBUG:
    logger.setLevel("DEBUG")
    ch.setLevel("DEBUG")
else:
    logger.setLevel("WARNING")
    ch.setLevel("WARNING")

logger.addHandler(ch)


def _getblock(M, block, blocksize):
    """ return the selected block of the given matrix
    """
    start = numpy.array(block) * blocksize

    B = M[start[0]:start[0] + blocksize,
          start[1]:start[1] + blocksize]

    logger.warning("index = %s", start)
    logger.debug("selected block: %s", B)
    return B


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

    @img.setter
    def img(self, value):
        self._img = value

    _color_image = False  # tells if the image is grayscale or color

    """ _key is a parameter which tells us how many blocks we select
        for ease of use we generate the blocks with the mersenne twister
        python RNG based on a small key that can be initialized"""
    _key = []

    _keysize = 8  # the number of selected blocks + some other fun stuff

    _blocks = 0  # the number of blocks (of _block_size) composing the image

    def __init__(self, imagefile, key, blocksize=8):

        self._block_size = blocksize
        self.img = cv2.imread(imagefile, cv2.CV_LOAD_IMAGE_UNCHANGED)
        # cv2 loads images in BGR. BGR -> RGB conversion needed

        if self.img.shape[2] == 3:
            self._color_image = True
            # it's a color image let's convert!
            # this way we can show it using pyplot

            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        self._blocks = (numpy.array(self.img.shape[:2]) /
                        self._block_size)
        h, w = self._blocks * self._block_size
        # since we are using 256x256 pixel images this step
        # shouldn't be needed but it is here just for completeness
        logger.debug("cropped image size")
        logger.debug("h,w: %s", (h, w))
        logger.debug("blocks: %s", self._blocks)

        self.img = self.img[:h, :w]

        self.initializeKey(key)

    def initializeKey(self, key):
        random.seed(key)  # initialize RNG

        # create the key selecting which blocks we are going to use
        self._key = zip(
            random.sample(range(0, self._blocks[0]), self._keysize),
            random.sample(range(0, self._blocks[1]), self._keysize))

        logger.debug("Key is: %s", self._key)

    def plotImg(self, name=None):

        pyplot.imshow(self.img)

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
            Y = numpy.zeros((_img.shape[0], _img.shape[1]), dtype=_img.dtype)
            Y[:, :] = _img[:, :, 0]

            # Cr = numpy.zeros((_img.shape[0], _img.shape[1]), dtype=_img.dtype)
            # Cr[:, :] = _img[:, :, 1]
            # Cb = numpy.zeros((_img.shape[0], _img.shape[1]), dtype=_img.dtype)
            # Cb[:, :] = _img[:, :, 2]

        else:
            Y = self.img

        logger.debug("Y: %s", Y)
        pyplot.figure('Y channel')
        pyplot.imshow(Y, cmap=pyplot.get_cmap('gray'))
        # pyplot.figure('Cr channel')
        # pyplot.imshow(Cr)
        # pyplot.figure('Cb channel')
        # pyplot.imshow(Cb)

        # now we use the selected block with the given key

        logger.debug("using blocks of size %d", self._block_size)

        for block in self._key:
            B = _getblock(Y, block, self._block_size)

    """        TransAll = []
            TransAllQuant = []
            ch = ['Y', 'Cr', 'Cb']
            plt.figure()
            for idx, channel in enumerate(imSub):
                plt.subplot(1, 3, idx + 1)
                channelrows = channel.shape[0]
                channelcols = channel.shape[1]
                Trans = np.zeros((channelrows, channelcols), np.float32)
                TransQuant = np.zeros((channelrows, channelcols), np.float32)
                blocksV = channelrows / B
                blocksH = channelcols / B
                vis0 = np.zeros((channelrows, channelcols), np.float32)
                vis0[:channelrows, :channelcols] = channel
                vis0 = vis0 - 128
                for row in range(blocksV):
                    for col in range(blocksH):
                        currentblock = cv2.dct(
                            vis0[row * B:(row + 1) * B, col * B:(col + 1) * B])
                        Trans[
                            row * B:(row + 1) * B, col * B:(col + 1) * B] = currentblock
                        TransQuant[
                            row * B:(row + 1) * B, col * B:(col + 1) * B] = np.round(currentblock / Q[idx])
                TransAll.append(Trans)
                TransAllQuant.append(TransQuant)
                if idx == 0:
                    selectedTrans = Trans[
                        srow * B:(srow + 1) * B, scol * B:(scol + 1) * B]
                else:
                    sr = np.floor(srow / SSV)
                    sc = np.floor(scol / SSV)
                    selectedTrans = Trans[sr * B:(sr + 1) * B, sc * B:(sc + 1) * B]
                plt.imshow(selectedTrans, cmap=cm.jet, interpolation='nearest')
                plt.colorbar(shrink=0.5)
                plt.title('DCT of ' + ch[idx])"""

    @property
    def is_color(self):
        return self._color_image


if __name__ == "__main__":
    # sf = SoftHash(
    #    './ImageDatabaseCrops/NikonD60/DS-01-UTFI-0196-0_crop.TIF',
    #    1234)
    sf = SoftHash('test.png', 1234)

    sf.update()

    # TODO: based on the keysize we can decide to resize the image
    # to match the final key size :)

    pyplot.show(block=True)


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
    + 1 int *R/G/B major component*

"""
