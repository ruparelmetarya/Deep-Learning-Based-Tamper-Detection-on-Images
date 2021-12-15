import os
import pandas as pd
from skimage.util import view_as_windows
import glob
import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable



def get_patches(image_mat, stride):
    """
    Extract patches rom an image
    :param image_mat: The image as a matrix
    :param stride: The stride of the patch extraction process
    :returns: The patches
    """
    window_shape = (128, 128, 3)
    print()
    windows = view_as_windows(np.array(image_mat), window_shape, step=stride)
    patches = []
    for m in range(windows.shape[0]):
        for n in range(windows.shape[1]):
            patches += [windows[m][n][0]]
    return patches


def get_images_and_labels(tampered_path, authentic_path):
    """
    Get the images and their corresponding labels
    :param tampered_path: The path containing the tampered images
    :param authentic_path: The path containing the authentic images
    :returns: Dictionary with images and labels
    """
    tampered_dir = tampered_path
    authentic_dir = authentic_path
    images = {}
    for im in glob.glob(authentic_dir):
        images[im] = {}
        images[im]['mat'] = cv2.imread(im)
        images[im]['label'] = 0
    for im in glob.glob(tampered_dir):
        images[im] = {}
        images[im]['mat'] = cv2.imread(im)
        images[im]['label'] = 1
    return images

def get_yi(model, patch):
    """
    Returns the patch's feature representation
    :param model: The pre-trained CNN object
    :param patch: The patch
    :returns: The 400-D feature representation of the patch
    """
    with torch.no_grad():
        model.eval()
        return model(patch)


def get_y_hat(y: np.ndarray, operation: str):
    """
    Fuses the image's patches feature representation
    :param y: The network object
    :param operation: Either max or mean for the pooling operation
    :returns: The final 400-D feature representation of the entire image
    """
    if operation == "max":
        return np.array(y).max(axis=0, initial=-math.inf)
    elif operation == "mean":
        return np.array(y).mean(axis=0)
    else:
        raise Warning("The operation can be either mean or max")


def create_feature_vectors(model, tampered_path, authentic_path, output_name):
    """
    Writes the feature vectors of the CASIA2 dataset.
    :param model: The pre-trained CNN object
    :param tampered_path: The path of the tampered images of the CASIA2 dataset
    :param authentic_path: The path of the authentic images of the CASIA2 dataset
    :param output_name: The name of the output CSV that contains the feature vectors
    """
    df = pd.DataFrame()
    images = get_images_and_labels(tampered_path, authentic_path)
    c = 1
    for image_name in images.keys():  # images
        print("Image: ", c)

        image = images[image_name]['mat']
        label = images[image_name]['label']

        df = pd.concat([df, pd.concat([pd.DataFrame([image_name.split(os.sep)[-1], str(label)]),
                                       pd.DataFrame(get_patch_yi(model, image))])], axis=1, sort=False)
        c += 1

    # save the feature vector to csv
    final_df = df.T
    final_df.columns = get_df_column_names()
    final_df.to_csv(output_name, index=False)  # csv type [im_name][label][f1,f2,...,fK]
    return final_df

def get_patch_yi(model, image):
    """
    Calculates the feature representation of an image.
    :param model: The pre-trained CNN object
    :param image: The image
    :returns: The image's feature representation
    """
    transform = transforms.Compose([transforms.ToTensor()])

    y = []  # init Y

    patches = get_patches(image, stride=1024)

    for patch in patches:  # for every patch
        img_tensor = transform(patch)
        img_tensor.unsqueeze_(0)
        img_variable = Variable(img_tensor.double())
        yi = get_yi(model=model, patch=img_variable)
        y.append(yi)  # append Yi to Y

    y = np.vstack(tuple(y))

    y_hat = get_y_hat(y=y, operation="mean")  # create Y_hat with mean or max

    return y_hat


def get_df_column_names():
    """
    Rename the feature csv column names as [im_names][labels][f1,f2,...,fK].
    :returns: The column names
    """
    names = ["image_names", "labels"]
    for i in range(400):
        names.append("f" + str(i + 1))
    return names

