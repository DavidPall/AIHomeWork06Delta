from PIL import Image
from random import randint
import numpy
from math import e
from matplotlib import image as mpimg
from pylab import *


def pandasVSdalmatas():
    train_test_ratio = 0.7
    lr = 0.00005
    train, test = load(train_test_ratio)

    print("len tr : {}    len ts : {}".format(len(train[0]), len(test[0])))
    w, e = OfflineLearning(train[0], train[1], v_sigmoid, v_gradf, lr, stop)

    res = decide(test[0], sigmoid, w)
    res = check_results(res)
    results = []
    for row in res:
        results.append(check_element(row))


    good_panda = 0
    bad_panda = 0
    good_dalmatian = 0
    bad_dalmatian = 0
    # print(results)
    for i in range(len(test[1])):
        if int(test[1][i][0]) == 1 and int(results[i]) == 1:
            good_panda += 1
        if int(test[1][i][0]) == 1 and int(results[i])== 0:
            bad_panda += 1
        if int(test[1][i][0]) == 0 and int(results[i]) == 0:
            good_dalmatian += 1
        if int(test[1][i][0]) == 0 and int(results[i]) == 1:
            bad_dalmatian += 1

    konf_matrix(good_panda, bad_panda, good_dalmatian, bad_dalmatian)

    output_list = []
    print_size = (6, 5)
    for i in range(print_size[0] * print_size[1]):
        if int(results[i]) == 1:
            decision = "panda"
        else:
            decision = "dalmata"
        output_list.append((test[2][i], decision))

    printPic(print_size[0], print_size[1], output_list)



def OfflineLearning(x, d, v_f, v_gradf, lr, stop):
    n = len(x[0])
    w = numpy.zeros((n, 1))
    for i in range(n):
        w[i][0] = (randint(0, 100) / 100)
    epoch = 0
    x = numpy.array(x, dtype=float)
    d = numpy.array(d, dtype=float)
    # print("x: {}".format(x[0]))
    # print("d: {}".format(d[0]))
    while True:
        v = x.dot(w)
        y = v_f(v)
        e = numpy.subtract(y, d)
        g = numpy.transpose(x)
        m = element_mult(e, v_gradf(v), len(x))
        g = g.dot(element_mult(e, v_gradf(v), len(x)))
        w = numpy.subtract(w,(lr*g))
        E = 0
        for i in e:
            E += i[0] ** 2
        if stop(epoch, E):
            break
        epoch += 1
    return w, E


def decide(x, fg, w):
    results = zeros((len(x), len(w)))
    for i in range(len(x)):
        for j in range(len(x[i])):
            q = x[i][j] + w[i][0]
            results[i][j] = fg(q)
    return results


def check_results(results):
    res = []
    for i in range(len(results)):
        res.append([])
        for j in range(len(results)):
            if results[i][j] > 0.5:
                res[i].append(1)
            else:
                res[i].append(0)
    return res


def check_element(list):
    sum = 0
    for element in list:
        sum += element
    if sum > len(list)/2:
        return 1 # panda
    else:
        return 0 # dalmatian


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
    scores = []
    for i in range(len(decals)):
        img = Image.open("std_images/std_" + str(i) + ".jpg", )
        sc = []
        for i in range(100):
            for j in range(100):
                sc.append(img.getpixel((i, j)) / 255)
        scores.append(sc)

    for i in range(len(decals)):
        img = Image.open("std_images/std_" + str(i) + ".jpg", )

        if i < cut_ind:
            x_train.append(scores[i])
            d_train.append([decals[i]])
            img_train.append(img)
        else:
            x_test.append(scores[i])
            d_test.append([decals[i]])
            img_test.append(img)

    print("x {}".format(shape(x_train)))

    return (x_train, d_train, img_train), (x_test, d_test, img_test)


def element_mult(v1, v2, ln):
    v3 = numpy.zeros((ln, 1))
    for i in range(ln):
        v3[i][0] = v1[i][0]*v2[i][0]
    return v3


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
    if e < 0.00001:
        return True
    return False


def printPic(n_rows, n_colums, output_list):
    for i in range(n_rows * n_colums):
        subplot(n_rows, n_colums, i + 1)
        title(output_list[i][1])
        #img = mpimg.imread("std_images/std_{}.jpg".format(output_list[i][0]))
        #plt.imshow(img)
        plt.imshow(output_list[i][0])
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