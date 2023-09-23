
"""
Imports we need.
Note: You may _NOT_ add any more imports than these.
"""
import argparse
import imageio
import logging
import numpy as np
from PIL import Image


def load_image(filename):
    """Loads the provided image file, and returns it as a numpy array."""
    im = Image.open(filename)
    return np.array(im)

def rgb2ycbcr(im):
    """Convert RGB to YCbCr."""
    xform = np.array([[0.299, 0.587, 0.114], [-0.1687, -0.3313, 0.5], [0.5, -0.4187, -0.0813]], dtype=np.float32)
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128.
    return ycbcr.astype(np.uint8)

def ycbcr2rgb(im):
    """Convert YCbCr to RGB."""
    xform = np.array([[1., 0., 1.402], [1, -0.34414, -0.71414], [1., 1.772, 0.]], dtype=np.float32)
    rgb = im.astype(np.float32)
    rgb[:,:,[1,2]] -= 128.
    rgb = rgb.dot(xform.T)
    return np.clip(rgb, 0., 255.).astype(np.uint8)

def create_gaussian_kernel(size, sigma=1.0):
    """
    Creates a 2-dimensional, size x size gaussian kernel.
    It is normalized such that the sum over all values = 1. 

    Args:
        size (int):     The dimensionality of the kernel. It should be odd.
        sigma (float):  The sigma value to use 

    Returns:
        A size x size floating point ndarray whose values are sampled from the multivariate gaussian.

    See:
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution
        https://homepages.inf.ed.ac.uk/rbf/HIPR2/eqns/eqngaus2.gif
    """

    # Ensure the parameter passed is odd
    if size % 2 != 1:
        raise ValueError('The size of the kernel should not be even.')

    # TODO: Create a size by size ndarray of type float32
    kernel = np.zeros((size, size), dtype=np.float32)

    # TODO: Populate the values of the kernel. Note that the middle `pixel` should be x = 0 and y = 0.
    for i in range(size):
        for j in range(size):
            x = j - size//2
            y = i - size//2
            # print(f'x = {x}, y = {y}')
            kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))

    # TODO:  Normalize the values such that the sum of the kernel = 1
    kernel = kernel / kernel.sum()

    return kernel.astype(np.float32)


def convolve_pixel(img, kernel, i, j):
    """
    Convolves the provided kernel with the image at location i,j, and returns the result.
    If the kernel stretches beyond the border of the image, it returns the original pixel.

    Args:
        img:        A 2-dimensional ndarray input image.
        kernel:     A 2-dimensional kernel to convolve with the image.
        i (int):    The row location to do the convolution at.
        j (int):    The column location to process.

    Returns:
        The result of convolving the provided kernel with the image at location i, j.
    """

    # First let's validate the inputs are the shape we expect...
    if len(img.shape) != 2:
        raise ValueError('Image argument to convolve_pixel should be one channel.')
    if len(kernel.shape) != 2:
        raise ValueError('The kernel should be two dimensional.')
    if kernel.shape[0] % 2 != 1 or kernel.shape[1] % 2 != 1:
        raise ValueError('The size of the kernel should not be even, but got shape %s' % (str(kernel.shape)))

    # TODO: determine, using the kernel shape, the ith and jth locations to start at.
    size = kernel.shape[0]
    # print('[size]', size)
    h, w = img.shape
    # print('[h, w]', h, w)
    # print('[i, j]', i, j)
    start_i = i - size // 2
    start_j = j - size // 2
    end_i = i + size // 2
    end_j = j + size // 2
    # print(f'start_i = {start_i}, start_j = {start_j}, end_i = {end_i}, end_j = {end_j}')

    convolved_pixel = 0
    # TODO: Check if the kernel stretches beyond the border of the image.
    if start_i < 0 or start_j < 0 or end_i >= h or end_j >= w:
        # TODO: if so, return the input pixel at that location.
        return img[i, j]
    else:
        # TODO: perform the convolution.
        # print('[img]\n', img)
        # print('[kernel]\n', kernel)
        for i2, y in enumerate(range(start_i, end_i + 1)):
            for j2, x in enumerate(range(start_j, end_j + 1)):
                # print(f'i2 = {i2}, j2 = {j2}, y = {y}, x = {x}, img = {img[y, x]}, kernel = {kernel[i2, j2]}')
                convolved_pixel = convolved_pixel + img[y, x] * kernel[i2, j2]
        # print('[convolved_pixel]', convolved_pixel)

    return convolved_pixel

def convolve(img, kernel):
    """
    Convolves the provided kernel with the provided image and returns the results.

    Args:
        img:        A 2-dimensional ndarray input image.
        kernel:     A 2-dimensional kernel to convolve with the image.

    Returns:
        The result of convolving the provided kernel with the image at location i, j.
    """
    # TODO: Make a copy of the input image to save results
    h, w = img.shape
    convolved_img = np.copy(img)

    # TODO: Populate each pixel in the input by calling convolve_pixel and return results.
    for i in range(h):
        for j in range(w):
            convolved_img[i, j] = convolve_pixel(img, kernel, i, j)
    # print('[img]\n', img)
    # print('[convolved_img]\n', convolved_img)
    
    return convolved_img

def split(img):
    """
    Splits a image (a height x width x 3 ndarray) into 3 ndarrays, 1 for each channel.

    Args:
        img:    A height x width x 3 channel ndarray.

    Returns:
        A 3-tuple of the r, g, and b channels.
    """
    if img.shape[2] != 3:
        raise ValueError('The split function requires a 3-channel input image')
    # TODO: Implement me
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    # print(r.shape, g.shape, b.shape)

    return (r, g, b)

def merge(r, g, b):
    """
    Merges three images (height x width ndarrays) into a 3-channel color image ndarrays.

    Args:
        r:    A height x width ndarray of red pixel values.
        g:    A height x width ndarray of green pixel values.
        b:    A height x width ndarray of blue pixel values.

    Returns:
        A height x width x 3 ndarray representing the color image.
    """
    # TODO: Implement me
    img = np.dstack((r, g, b))
    # print('[img.shape]', img.shape)

    return img


"""
The main function
"""
if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(description='Blurs an image using an isotropic Gaussian kernel.')
    parser.add_argument('input', type=str, help='The input image file to blur')
    parser.add_argument('output', type=str, help='Where to save the result')
    parser.add_argument('--ycbcr', action='store_true', help='Filter in YCbCr space')
    parser.add_argument('--sigma', type=float, default=1.0, help='The standard deviation to use for the Guassian kernel')
    parser.add_argument('--k', type=int, default=5, help='The size of the kernel.')
    parser.add_argument('--subsample', type=int, default=1, help='Subsample by factor')
    
    # python3 hw1.py example_1.input.jpg example_1.output.jpg
    # python3 hw1.py example_1.input.jpg example_1.output-ycbcr.jpg --ycbcr
    # python3 hw1.py example_1.input.jpg example_1.output-sigma_5.jpg --sigma 5
    # python3 hw1.py example_1.input.jpg example_1.output-k_9.jpg --k 9
    # python3 hw1.py example_1.input.jpg example_1.output-sigma_5-k_9.jpg --sigma 5 --k 9
    # python3 hw1.py example_1.input.jpg example_1.output-subsample_2.jpg --subsample 2

    # python3 hw1.py example_2.input.jpg example_2.output.jpg
    # python3 hw1.py example_2.input.jpg example_2.output-ycbcr.jpg --ycbcr
    # python3 hw1.py example_2.input.jpg example_2.output-sigma_5.jpg --sigma 5
    # python3 hw1.py example_2.input.jpg example_2.output-k_9.jpg --k 9
    # python3 hw1.py example_2.input.jpg example_2.output-sigma_5-k_9.jpg --sigma 5 --k 9
    # python3 hw1.py example_2.input.jpg example_2.output-subsample_2.jpg --subsample 2

    args = parser.parse_args()

    # first load the input image
    logging.info('Loading input image %s' % (args.input))
    inputImage = load_image(args.input)

    if args.ycbcr:
        # Convert to YCbCr
        inputImage = rgb2ycbcr(inputImage)

        # Split it into three channels
        logging.info('Splitting it into 3 channels')
        (y, cb, cr) = split(inputImage)

        # compute the gaussian kernel
        logging.info('Computing a gaussian kernel with size %d and sigma %f' % (args.k, args.sigma))
        kernel = create_gaussian_kernel(args.k, args.sigma)

        # convolve it with cb and cr
        logging.info('Convolving the Cb channel')
        cb = convolve(cb, kernel)
        logging.info('Convolving the Cr channel')
        cr = convolve(cr, kernel)

        # merge the channels back
        logging.info('Merging results')
        resultImage = merge(y, cb, cr)

        # convert to RGB
        resultImage = ycbcr2rgb(resultImage)
    else:
        # Split it into three channels
        logging.info('Splitting it into 3 channels')
        (r, g, b) = split(inputImage)

        # compute the gaussian kernel
        logging.info('Computing a gaussian kernel with size %d and sigma %f' % (args.k, args.sigma))
        kernel = create_gaussian_kernel(args.k, args.sigma)

        # convolve it with each input channel
        logging.info('Convolving the first channel')
        r = convolve(r, kernel)
        logging.info('Convolving the second channel')
        g = convolve(g, kernel)
        logging.info('Convolving the third channel')
        b = convolve(b, kernel)

        # merge the channels back
        logging.info('Merging results')
        resultImage = merge(r, g, b)

    # subsample image
    if args.subsample != 1:
        # subsample by a factor of 2
        resultImage = resultImage[::args.subsample, ::args.subsample, :]

    # save the result
    logging.info('Saving result to %s' % (args.output))
    imageio.imwrite(args.output, resultImage)
