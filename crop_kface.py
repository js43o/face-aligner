import os
import cv2
from aligner import align_image

dataroot = "../../datasets/kface"
new_dataroot = "../../datasets/kface_crop"
classes = ["train", "test"]

for cls in classes:
    ids = os.listdir(os.path.join(dataroot, cls))
    ids.sort()
    for idx, id in enumerate(ids):
        print("🔥 processing %s:%d ..." % (cls, idx))
        for light in range(1, 21):
            for expression in range(1, 4):
                filepath = os.path.join(
                    dataroot, cls, id, "S001", "L%s" % light, "E0%s" % expression
                )
                savepath = os.path.join(
                    new_dataroot, cls, id, "S001", "L%s" % light, "E0%s" % expression
                )
                os.makedirs(savepath, exist_ok=True)

                for angle in range(1, 21):
                    filename = os.path.join(filepath, "C%s.jpg" % angle)
                    image = cv2.imread(filename)

                    aligned = align_image(image)
                    if aligned is None or aligned.size == 0:
                        continue

                    cv2.imwrite(os.path.join(savepath, "C%s.png" % angle), aligned)
