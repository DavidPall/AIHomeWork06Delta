from PIL import Image
from random import randint
import numpy
import scipy
from scipy import stats
from math import e
from matplotlib import image as mpimg
from pylab import *


def OfflineLearning(x, d, v_f, v_gradf, lr, stop):
    n = len(x[0])
    w = numpy.zeros((n, 1))
    for i in range(n):
        w[i][0] = (randint(0, 100) / 100)
    epoch = 0
    x = numpy.array(x, dtype=float)
    d = numpy.array(d, dtype=float)
    while True:
        v = x.dot(w)
        y = v_f(v)
        e = numpy.subtract(y, d)
        g = numpy.transpose(x).dot(e.dot(v_gradf(v)))
        w = w.subtract(lr * g)
        E = 0
        for i in e:
            E += i ** 2
        if stop(epoch, E):
            break
        epoch += 1
    return w, E


def load(ratio):
    decals = []
    with open("std_images/decals.txt") as f:
        for line in f:
            line = line.split()
            decals.append(line[0])

    cut_ind = len(decals) * ratio

    x_train = []
    x_test = []
    d_train = []
    d_test = []
    img_train = []
    img_test = []
    for i in range(len(decals)):
        img = Image.open("std_images/std_" + str(i) + ".jpg", )
        z_score = scipy.stats.zscore(img)
        f.close()
        if i < cut_ind:
            x_train.append(z_score)
            d_train.append([decals[i]])
            img_train.append(img)
        else:
            x_test.append(z_score)
            d_test.append([decals[i]])
            img_test.append(img)

    return (x_train, d_train, img_train), (x_test, d_test, img_test)


def sigmoid(x):
    return 1 / (1 + e ** (-x))


def gradf(x):
    return sigmoid(x) * (1 - sigmoid(x))


def v_sigmoid(x):
    for i in range(len(x)):
        x[i][0] = sigmoid(x[i][0])
    return numpy.array(x, dtype=float)


def v_gradf(x):
    for i in range(len(x)):
        x[i][0] = gradf(x[i][0])
    return numpy.array(x, dtype=float)


def stop(n, e):
    if n > 5000:
        return True
    if e < 0.001:
        return True
    return False


def pandasVSdalmatas():
    train_test_ratio = 0.7
    lr = 0.0005
    train, test = load(train_test_ratio)

    print("len tr : {}    len ts : {}".format(len(train[0]), len(test[0])))
    w, e = OfflineLearning(train[0], train[1], v_sigmoid, v_gradf, lr, stop)


def printPic(n_rows, n_colums, output_list):
    for i in range(n_rows * n_colums):
        subplot(n_rows, n_colums, i + 1)
        title(output_list[i][1])
        img = mpimg.imread("std_images/std_{}.jpg".format(output_list[i][0]))
        plt.imshow(img)
        plt.axis('off')
        plt.subplots_adjust(left=0, bottom=0, right=1, top=0.94, wspace=0.20, hspace=0.48)

    show()


def konf_matrix(good_pan, bad_pan, good_dalm, bad_dalm):
    values = [['My_Pandas', good_pan, bad_pan], ['My_Dalmatians', bad_dalm, good_dalm]]
    collabel = ("", "Pandas", "Dalmatians")
    fig, axs = plt.subplots(2, 1)
    colours = [['white', 'lime', 'red'], ['white', 'red', 'lime']]
    axs[0].axis('off')
    the_table = axs[0].table(cellText=values, colLabels=collabel, colLoc='center', rowLoc='center', cellColours=colours)
    plt.subplots_adjust(left=0.25, right=0.80)
    plt.axis('off')
    plt.show()


pandasVSdalmatas()