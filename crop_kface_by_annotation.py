import os
from PIL import Image

srcdir = "samples"
outdir = "out"
classes = ["train", "val", "test"]

for angle in range(1, 21):
    image = Image.open(os.path.join(srcdir, "C%s.jpg" % angle)).convert("RGB")
    meta = open(os.path.join(srcdir, "C%s.txt" % angle), "r").readlines()

    left, top, width, height = map(int, meta[7].split("\t"))
    image = image.crop((left, top, left + width, top + height))

    image.save(os.path.join(outdir, "C%s.jpg" % angle))
