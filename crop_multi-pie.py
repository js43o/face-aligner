import os
import cv2
from aligner import align_image

dataroot = "../../datasets/multi-pie"
new_dataroot = "../../datasets/multi-pie_crop_patch"
classes = ["test"]
ANGLES = [
    "11_0",
    "12_0",
    "09_0",
    "08_0",
    "13_0",
    "14_0",
    "05_1",
    "05_0",
    "04_1",
    "19_0",
    "20_0",
    "01_0",
    "24_0",
    "08_1",
    "19_1",
]
SESSION = "01"
RECORDING = "01"


for cls in classes:
    ids = os.listdir(os.path.join(dataroot, cls))
    ids.sort()
    for idx, id in enumerate(ids):
        print("🔥 processing %s:%s (%d/%d)..." % (cls, id, idx + 1, len(ids)))
        for angle in ANGLES:
            savepath = os.path.join(new_dataroot, cls, id, RECORDING, angle)
            os.makedirs(savepath, exist_ok=True)

            for light in range(20):
                filename = "%s_%s_%s_%s_%s.png" % (
                    id,
                    SESSION,
                    RECORDING,
                    angle.replace("_", ""),
                    "%02d" % light,
                )
                filepath = os.path.join(dataroot, cls, id, RECORDING, angle, filename)
                image = cv2.imread(filepath)

                if angle in ("08_1", "19_1"):
                    image = cv2.flip(image, 0)

                aligned = align_image(image)

                if aligned is None or aligned.size == 0:
                    continue

                cv2.imwrite(os.path.join(savepath, "%02d.png" % light), aligned)
