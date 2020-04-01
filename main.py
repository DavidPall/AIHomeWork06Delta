from PIL import Image
import random
import numpy
import scipy
from scipy import stats


def OfflineLearning(x, d, v_f, v_gradf, lr, stop):
    n = len(x[0])
    w = []
    for i in range(n):
        w.append([(random.randint % 100) / 100])
    epoch = 0
    while True:
        v = x * w
        y = v_f(v)
        e = y - d
        g = numpy.transpose(x) * (e * v_gradf(v))
        w = w - lr * g
        E = 0
        for i in e:
            E += i**2
        if stop(E, epoch):
            break
        epoch += 1
    return w, E



def load(ratio):
    decals = []
    with open("std_images\\decals.txt") as f:
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
        img = Image.open("std_images\\std_" + str(i) + ".jpg", )
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


train, test = load(0.7)

print("len tr : {}    len ts : {}".format(len(train), len(test)))