import PIL
from PIL import Image
import random

list_images = []
decals = [] # 0 == dalmatian   1 == panda
with open("dalmatian\\list.txt") as f:
    for line in f:
        line = line.split('\n')
        img = Image.open("dalmatian\\" + line[0])
        # hpercent = (baseheight / float(img.size[1]))
        # wsize = int((float(img.size[0]) * float(hpercent)))
        img = img.resize((100, 100), PIL.Image.ANTIALIAS)
        img = img.convert('1')
        # img.save(‘resized_image.jpg')
        list_images.append(img)
        decals.append(0)

with open("panda\\list.txt") as f:
    for line in f:
        line = line.split('\n')
        img = Image.open("panda\\" + line[0])
        # hpercent = (baseheight / float(img.size[1]))
        # wsize = int((float(img.size[0]) * float(hpercent)))
        img = img.resize(((100, 100)), PIL.Image.ANTIALIAS)
        img = img.convert('1')
        # img.save(‘resized_image.jpg')
        list_images.append(img)
        decals.append(1)

order = []
for i in range(len(list_images)):
    order.append(i)

random.shuffle(order)

decal_output = []
for i in range(len(list_images)):
    list_images[order[i]].save("std_images\\std_" + str(i) + ".jpg")
    decal_output.append(decals[order[i]])


fout = open("std_images\\decals.txt", "w")
for d in decal_output:
    fout.write(str(d) + '\n')

fout.close()