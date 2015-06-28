import cv2
import numpy
from matplotlib import pyplot, cm


import matplotlib
matplotlib.interactive(True)


DEBUG = True


class SoftHash(object):

    _block_size = 8     # use an 8x8 blocksize for the DCT
    _img = None    # the original image we are working on
    _img_cropped = None  # the cropped image since we work on 8x8 blocks

    def __init__(self, imagefile, blocksize=8):

        self._block_size = blocksize
        self._img = cv2.imread(imagefile, cv2.CV_LOAD_IMAGE_UNCHANGED)
        # cv2 loads images in BGR. BGR -> RGB conversion needed
        if self._img.shape[2] == 3:
            # it's a color image let's convert! :)
            self._img = cv2.cvtColor(self._img, cv2.COLOR_BGR2RGB)

        """
        instead of cv2 PIL can be used too and conversion is not needed
        from PIL import Image
        im = Image.open('a_image.tif')
        im.show()
        """

        pyplot.imshow(self._img)
        raw_input()

        h, w = (numpy.array(self._img.shape[:2]) /
                self._block_size) * self._block_size
        # since we are using 256x256 pixel images this step
        # shouldn't be needed but it is here just for completeness

        self._img_cropped = self._img[:h, :w]
        pyplot.imshow(self._img)
        raw_input()

    def plotOriginalImg(self):
        pyplot.imshow(self._img)


if __name__ == "__main__":
    #sf = SoftHash('./ImageDatabaseCrops/NikonD60/DS-01-UTFI-0000-0_crop.TIF')
    sf = SoftHash('Ub7XL8T.png')
    # sf.plotOriginalImg()
