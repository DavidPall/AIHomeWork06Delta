from math import e
from matplotlib import image as mpimg
from pylab import *
import plotly.graph_objects as go
import plotly.io as pio

def sigmoid(x):
    return 1 / (1 + e ** (-x))


def gradf(x):
    return sigmoid(x) * (1 - sigmoid(x))

def v_sigmoid(x):
    for i in range(len(x)):
        x[i][0] = sigmoid(x[i][0])
    return x

def v_gradf(x):
    for i in range(len(x)):
        x[i][0] = gradf(x[i][0])
    return x


def pandasVSdalmatas():
    train_test_ratio = 0.7
    lr = 0.0005


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

    values = [['My_Pandas', good_pan, bad_pan],['My_Dalmatians', bad_dalm, good_dalm]]
    collabel = ("", "Pandas", "Dalmatians")
    fig, axs = plt.subplots(2,1)
    colours = [['white','lime','red'],['white','red','lime']]
    axs[0].axis('off')
    the_table = axs[0].table(cellText=values, colLabels=collabel, colLoc='center', rowLoc='center', cellColours=colours)
    plt.subplots_adjust(left=0.25, right=0.80)
    plt.axis('off')
    plt.show()

