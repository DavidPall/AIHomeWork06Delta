import PIL
from PIL import Image
import random

list_images = []
with open("dalmatian\\list.txt") as f:
    for line in f:
        line = line.split('\n')
        img = Image.open("dalmatian\\" + line[0])
        # hpercent = (baseheight / float(img.size[1]))
        # wsize = int((float(img.size[0]) * float(hpercent)))
        img = img.resize((500, 500), PIL.Image.ANTIALIAS)
        img = img.convert('1')
        # img.save(‘resized_image.jpg')
        list_images.append(img)

with open("panda\\list.txt") as f:
    for line in f:
        line = line.split('\n')
        img = Image.open("panda\\" + line[0])
        # hpercent = (baseheight / float(img.size[1]))
        # wsize = int((float(img.size[0]) * float(hpercent)))
        img = img.resize(((500, 500)), PIL.Image.ANTIALIAS)
        img = img.convert('1')
        # img.save(‘resized_image.jpg')
        list_images.append(img)

order = []
for i in range(len(list_images)):
    order.append(i)

random.shuffle(order)

for i in range(len(list_images)):
    list_images[order[i]].save("std_images\\std_" + str(i) + ".jpg")