from math import e
from matplotlib import image as mpimg
from pylab import *



def sigmoid(x):
    return 1 / (1 + e**(-x))


def gradf(x):
    return sigmoid(x) * (1 - sigmoid(x))


def pandasVSdalmatas():

    train_test_ratio = 0.7
    lr = 0.0005


def print(n_rows, n_colums, output_list):

    for i in range(n_rows * n_colums):
        subplot(n_rows, n_colums, i + 1)
        title(output_list[i][1])
        img = mpimg.imread("std_images/std_{}.jpg".format(output_list[i][0]))
        plt.imshow(img)
        plt.axis('off')
        plt.subplots_adjust(left=0, bottom=0, right=1, top=0.94, wspace=0.20, hspace=0.48)

    show()
