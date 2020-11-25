import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import img_as_float
from scipy import ndimage


def order_func(x):
    name = os.path.splitext(x)[0]
    return int(name[:-2] + str(ord(name[-2]) % 97))


def obtain_data(data_dir="../data/vessels"):
    train_fnames = sorted(os.listdir(data_dir + '/train/images'), key=order_func)
    train_imgs = img_as_float(np.array([np.array(Image.open(data_dir + "/train/images/" + f)) for f in train_fnames]))
    train_labels = img_as_float(np.array([np.load(data_dir + "/train/gt/" + f + ".npy") for f in train_fnames]))

    test_fnames = sorted(os.listdir(data_dir + '/test/images'), key=order_func)
    test_imgs = img_as_float(np.array([np.array(Image.open(data_dir + "/test/images/" + f)) for f in test_fnames]))
    test_labels = img_as_float(np.array([np.load(data_dir + "/test/gt/" + f + ".npy") for f in test_fnames]))

    return {"train": {"names": train_fnames, "images": train_imgs, "labels": train_labels},
            "test":  {"names": test_fnames, "images": test_imgs, "labels": test_labels}}


def scale_to_uint(image):
    return ((image - image.min()) * (1 / (image.max() - image.min()) * 255)).astype('uint8')


def lamdafind(hess_eles):
    hess = np.array(hess_eles).reshape(2, 2)
    lam1, lam2 = np.linalg.eig(hess)[0]
    if lam1 < lam2:
        return lam2
    return lam1


def find_start_point(image, fs=20):
    image = scale_to_uint(image)
    rows, cols = image.shape
    best_i, best_j = None, None
    best_min = np.inf
    for i in range(rows-fs):
        for j in range(cols-fs):
            curr_min = np.sum(image[i:i+fs, j:j+fs])
            if curr_min < best_min:
                best_min = curr_min
                best_i = i
                best_j = j
    return (best_j+(fs/2))/cols, (best_i+(fs/2))/rows


def principal_curvature(image):
    rows, cols = image.shape
    gx = ndimage.sobel(image, axis=0, mode='constant')
    gy = ndimage.sobel(image, axis=1, mode='constant')

    gxx = ndimage.sobel(gx, axis=0, mode='constant')
    gxy = ndimage.sobel(gx, axis=1, mode='constant')  # same as gyx

    gyy = ndimage.sobel(gy, axis=1, mode='constant')

    lamdaplus = np.zeros(image.shape)

    for i in range(rows):
        for j in range(cols):
            lamdaplus[i, j] = lamdafind([gxx[i, j], gxy[i, j], gxy[i, j], gyy[i, j]])

    return img_as_float(lamdaplus)


# A function that runs the level set method with a given image.
def run_lsm(image, lsm, step_size=1, max_iter=10, print_every=10, do_plot=True):
    for outer_iter in range(max_iter):
        # initialize phi
        lsm.phi[lsm.phi < 0] = -lsm.rho
        lsm.phi[lsm.phi >= 0] = lsm.rho

        # apply heat equation
        lsm.phi = lsm.conv(lsm.phi)

        # compute f1 and f2
        lsm.compute_local_bin_value(image)

        # compute c1 and c2
        lsm.compute_the_region_average_intensity(image)

        # Compute Spf
        lsm.compute_the_local_global_force(image)

        # compute |\nabla phi|
        lsm.compute_absolute_gradient()

        # update phi
        lsm.update_phi(step_size, image)

        # print iteration and show images if do_plot == True
        if (outer_iter % print_every == 0):
            if do_plot:
                print("Iter: {:3d}".format(outer_iter))
                plt.cla()
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].imshow(image, 'bone')
                ax[0].set_title("Image")
                ax[1].contour(np.flipud(lsm.phi))
                ax[1].set_title("iter: {:3d}".format(outer_iter))
                plt.tight_layout()
                plt.pause(0.01)

        plt.close()
    return lsm.phi
