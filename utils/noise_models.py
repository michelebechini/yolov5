import cv2
import numpy as np

def speed_noise(image, seed=None, clip=True):
    """
    Function to apply to the input image the noise given prescribed for the SPEED dataset.


    :param image: ndarray <br>
                  Will be converted to float. <br> <br>

    :param seed: int, optional <br>
                 If provided, this will set the random seed before generating noises, for valid pseudo-random
                 comparisons. <br> <br>

    :param clip: bool, optional <br>
                 If TRUE (default), the output will be clipped after noise applied. This is needed to maintain the the
                 proper image data range. If FALSE the output may extend beyond the correct range. <br> <br>

    :return: out: ndarray <br>
                  Output image data. <br> <br>
    """

    if image.max() > 1:
        img_norm = image / 255.
    else:
        img_norm = image.copy()

    # apply BLUR
    noised_1_norm = cv2.GaussianBlur(img_norm, ksize=(0, 0), sigmaX=1., sigmaY=0.)
    row, col = noised_1_norm.shape
    mean = 0
    var = 0.0022
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noised_2_norm = noised_1_norm + gauss
    noised_2_norm = np.clip(noised_2_norm, 0, 1)

    # bring the normalized noised image in the [0, 1] range
    #noised_2_norm = cv2.normalize(noised_2_norm.astype('float64'), None, alpha=0., beta=1.,
    #                              norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


    if image.max() > 1:
        return noised_2_norm * 255
    else:
        return noised_2_norm

def blur_image(image, ksize=0, sigma=1.15):
    """
    Function to apply gaussian blurring to an image

    :param image: ndarray <br>
                  Will be converted to float. <br> <br>

    :param ksize: int <br>
                  Kernel size for gaussian blur. Default ksize = 0.

    :param sigma: float <br>
                  Sigma value for Gaussian blur. Default sigma = 1.15

    :return: out: ndarray <br>
                  Output image data. <br> <br>
    """

    filtered_img = cv2.GaussianBlur(image.astype('float64'), ksize=(ksize, ksize), sigmaX=sigma,
                                    borderType=cv2.BORDER_REPLICATE)

    return filtered_img
