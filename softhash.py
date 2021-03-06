#!/usr/bin/python

import cv2
import numpy as np
from matplotlib import pyplot as plt
import random

import logging
from glob import glob
import matplotlib

import bitstring

from noise import noisy

matplotlib.interactive(True)
matplotlib.use("TkAgg")
# plt.ion()

import PIL
from StringIO import StringIO

DEBUG = False

logger = logging.getLogger("softhash")
ch = logging.StreamHandler()

if not DEBUG:
    logger.setLevel("WARNING")
    ch.setLevel("WARNING")
else:
    logger.setLevel("DEBUG")
    ch.setLevel("DEBUG")

logger.addHandler(ch)


class ImageHash(object):

    """
    Hash encapsulation. Can be used for dictionary keys and comparisons.
    """

    def __init__(self, binary_array):
        self.hash = binary_array

    def __str__(self):
        return "%s" % self.hash.flatten()

    def __repr__(self):
        return repr(self.hash)

    def __sub__(self, other):
        if other is None:
            raise TypeError('Other hash must not be None.')
        if self.hash.size != other.hash.size:
            raise TypeError(
                'ImageHashes must be of the same shape.',
                self.hash.shape, other.hash.shape)
        return (self.hash.flatten() != other.hash.flatten()).sum()

    def __eq__(self, other):
        if other is None:
            return False
        return np.array_equal(self.hash.flatten(), other.hash.flatten())

    def __ne__(self, other):
        if other is None:
            return False
        return not np.array_equal(self.hash.flatten(), other.hash.flatten())

    def __hash__(self):
        # this returns a 8 bit integer, intentionally shortening the
        # information
        return sum(
            [2**(i % 8) for i, v in enumerate(self.hash.flatten()) if v])


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

    def __init__(self,
                 imagefile,
                 key,
                 blocksize=16,
                 selectedblocks=16,
                 maskfactor=10,
                 resize=(64, 64),
                 ):
        self._key = []
        self._block_size = blocksize

        if type(imagefile) is np.ndarray:
            self.img = imagefile
        else:
            self.img = cv2.imread(imagefile, cv2.CV_LOAD_IMAGE_UNCHANGED)

        # by default we denoise the image using non-local means algorithm
        # should denoising be done before or after resizing?
        self.denoise()

        # we have a DB of 256x256 cropped images, subsample
        self.img = cv2.resize(self.img, resize)
        # cv2 loads images in BGR. BGR -> RGB conversion needed

        if self.img.shape[2] == 3:
            self._color_image = True
            # it's a color image let's convert to RGB!
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        self._blocks = np.array(self.img.shape[:2]) / self._block_size

        h, w = self._blocks * self._block_size
        # since we are using 256x256 pixel images this step
        # shouldn't be needed but it is here just for completeness
        logger.debug("cropped image size")
        logger.debug("h, w: %s", (h, w))
        logger.debug("blocks: %s", self._blocks)

        self.img = self.img[:h, :w]

        # ensure that we select at most the
        # number of blocks composing the image
        if selectedblocks is None:
            selectedblocks = self._nblocks
        elif selectedblocks > self._nblocks:
            selectedblocks = self._nblocks

        self._keysize = selectedblocks

        self.initializeKey(key)

        self.mask = np.rot90(
            np.triu(np.ones([blocksize, blocksize]), maskfactor))

        self._hash()

    def initializeKey(self, key):
        random.seed(key)  # initialize RNG

        # create the key selecting which blocks we are going to use
        selectedblocks = random.sample(range(0, self._nblocks), self._keysize)

        # get the block element in the matrix
        for block in selectedblocks:
            x = block % self._blocks[0]
            y = int(np.floor(block / self._blocks[0]))
            self._key.append((x, y))

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

    def _hash(self):

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

        # now we use the selected block with the given key
        logger.debug("using blocks of size %d\n", self._block_size)

        if DEBUG:
            # needed to calc the square layout for plots
            sqr = np.ceil(np.sqrt(self._keysize))

        # logger.debug("mask matrix: %s\n", self.mask)

        transformedBlocks = []  # just keep all the blocks somewhere
        self.coeffs = []

        for idx, block in enumerate(self.key_pixels):

            # now like the JPEG standard we shift the values by -128
            # don't know if it would change anything with cv2 dct
            # implementation... but looks like a good rule to follow
            # http://compgroups.net/comp.compression/level-shift-in-jpeg-optional-or-mandatory/175097

            B = np.zeros((blocksize, blocksize), dtype=np.int8)

            B[:, :] = Y[
                block[0]:block[0] + blocksize,
                block[1]:block[1] + blocksize] - 128

            # logger.debug("selected block index = %s", block)
            # logger.debug("selected block: %s", B)

            Bdct = cv2.dct(np.array(B, dtype=np.float32))
            # logger.debug("block dct: %s", Bdct)

            # quantize the block DCT cleaning out high frequencies
            Qdct = np.array(np.multiply(self.mask, Bdct))

            # logger.debug("quantized block DCT: %s", Qdct)

            if DEBUG:
                # show the selected blocks

                plt.figure('selected blocks')
                plt.subplot(sqr, sqr, idx + 1)
                plt.imshow(
                    B+128, cmap=plt.get_cmap('gray'),
                    interpolation='nearest')

                plt.figure('blocks DCT')
                plt.subplot(sqr, sqr, idx + 1)
                plt.imshow(
                    Bdct, cmap=plt.get_cmap('jet'),
                    interpolation='nearest')

                # inverse DCT and shift again +128 to check results
                # an int 16 should be enough to store results of the DCT
                # (even less bits could be used probably!)
                invBdct = np.array(
                    cv2.idct(Qdct)+128, dtype=np.uint8)

                plt.figure('decoded DCT blocks')
                plt.subplot(sqr, sqr, idx + 1)
                plt.imshow(
                    invBdct, cmap=plt.get_cmap('gray'),
                    interpolation='nearest')
            transformedBlocks.append(Qdct)

            # let's see what are the indices of the coeffs
            # we start from the mask we defined since it will
            # help us keep a constant number of coeffs
            # which means we can easily compare images
            # using a hamming distance
            indices = np.transpose(self.mask.nonzero())

            # get indices for coeffs (keep in mind we don't want the DC coeff!
            for idx in indices:
                if not idx.any():
                    continue        # skip the DC component
                self.coeffs.append(Qdct[idx[0]][idx[1]])

        # coeffs now contains the coefficients
        # we are interested in inserting into the hash
        # now let's quantize them

        logger.debug("using %d coeffs\n", len(self.coeffs))
        # now quantize and return values
        return True

    def hexdigest(self, hashsize=1024):
        hashBitsPerCoeff = (
            float(hashsize) / len(self.coeffs))
        hashBitsPerCoeff = int(np.ceil(hashBitsPerCoeff))

        hashbitsround = len(self.coeffs) * hashBitsPerCoeff
        if hashbitsround != hashsize:
            logger.warning(
                "hash size was rounded up to %d bits",
                hashbitsround)
            # we will use some more bits who cares! :]
        self.hashBitsPerCoeff = hashBitsPerCoeff

        return self._quantizehash()

    def _quantizehash(self):
        """quantize the hash and return the values"""

        def quantize_coeff(x, xMax, mu):
            qDiv = 2*xMax >> (hashBitsPerCoeff)

            y = x
            if y >= xMax:
                logger.warning("%f has been capped!",  y)
                y = xMax  # cap numbers too big
            # use mu law for non uniform quantization
            y = np.log(1 + (mu * (np.absolute(y)/xMax)))
            y *= xMax
            y /= np.log(1 + mu)
            y *= np.sign(x)

            y += xMax  # shift to have positive values!
            y /= qDiv

            y = int(y) - 1
            if y < 0:
                y = 0
            return y

        computed_hash = bitstring.BitArray()
        hashBitsPerCoeff = self.hashBitsPerCoeff

        # get the MaxDCT value possible
        MaxDCTVal = 255 * self._block_size * self._block_size
        # cap the max values mu law still not good enough
        MaxDCTVal = 1024

        # compute average this way we can quantize better
        # avg_coeff = np.average(self.coeffs)
        # since we use a non linear quant. no average shift...
        mu = 250
        for k in self.coeffs:
            y = quantize_coeff(k, MaxDCTVal, mu)

            computed_hash.append(
                bitstring.Bits(uint=y, length=hashBitsPerCoeff))

        self._softhash = computed_hash
        return computed_hash

    def hamming_distance(self, b):
        return (self._softhash ^ b).count(True)

    def similarity(self, b):
        d = self.hamming_distance(b)
        return 100 - ((d * 100) / self._softhash.len)

    @property
    def is_color(self):
        return self._color_image


def hamming_distance(a, b):
    return (a ^ b).count(True)


def buff_to_cvimg(img_stream, cv2_img_flag=0):
    img_stream.seek(0)
    img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)

if __name__ == "__main__":

    key = 1234
    f = './ImageDatabaseCrops/NikonD60/DS-01-UTFI-0169-0_crop.TIF'
    # f = 'test.png'
    sf = SoftHash(f, key)

    h = sf.hexdigest()

    f = './ImageDatabaseCrops/NikonD3000/DSC_0189_crop.TIF'
    f = './ImageDatabaseCrops/NikonD200_D2/Nikon_D200_1_17215_crop.TIF'
    sf2 = SoftHash(f, key)

    h = sf2.hexdigest()
    print sf.similarity(h)

    # coeffs = []
    # for f in glob('./ImageDatabaseCrops/*/*.TIF'):
    #    h = SoftHash(f, key)
    #    coeffs.append(h.coeffs)

    # plt.hist(coeffs, 50, normed=1, facecolor='b', alpha=0.75)

    # TESTS

    N = 0
    results = {
        'blur': {},
        'jpeg': {},
        'noise': {}
    }

    keysizes = [k*320 for k in range(1, 10)]

    for keysize in keysizes:
        results['blur'][keysize] = 0
        results['jpeg'][keysize] = 0
        results['noise'][keysize] = 0

    for f in glob('./ImageDatabaseCrops/NikonD200_D2/*.TIF'):
        img_orig = cv2.imread(f, cv2.CV_LOAD_IMAGE_UNCHANGED)
        # Gaussian Blur (low pass filtering)
        img_blur = cv2.blur(img_orig, (5, 5))
        # JPEG compression

        buff = StringIO()
        im1 = PIL.Image.open(f)
        im1.save(buff, "JPEG", quality=30)
        img_jpeg = buff_to_cvimg(buff, cv2.CV_LOAD_IMAGE_UNCHANGED)

        # noise addition
        img_noisy = noisy('speckle', img_orig)

        sho = SoftHash(img_orig, key)

        shb = SoftHash(img_blur, key)
        shj = SoftHash(img_jpeg, key)
        shn = SoftHash(img_noisy, key)

        for keysize in keysizes:
            ho = sho.hexdigest(keysize)
            hb = shb.hexdigest(keysize)
            hj = shb.hexdigest(keysize)
            hn = shn.hexdigest(keysize)

            results['blur'][keysize] += hamming_distance(ho, hb)
            results['jpeg'][keysize] += hamming_distance(ho, hj)
            results['noise'][keysize] += hamming_distance(ho, hn)

        N += 1

    blur_errs = []
    jpeg_errs = []
    noise_errs = []

    print keysizes
    print N
    for keysize in keysizes:
        blur_errs.append(results['blur'][keysize] * 100 / N / keysize)
        jpeg_errs.append(results['jpeg'][keysize] * 100 / N / keysize)
        noise_errs.append(results['noise'][keysize] * 100 / N / keysize)

    plt.figure('blur errors')
    plt.plot(keysizes, blur_errs, 'k', keysizes, blur_errs, 'ro')

    plt.figure('jpeg errors')
    plt.plot(keysizes, jpeg_errs, 'k', keysizes, jpeg_errs, 'ro')

    plt.figure('noise errors')
    plt.plot(keysizes, noise_errs, 'k', keysizes, noise_errs, 'ro')

    plt.show(block=True)
