import os
from PIL import Image
import numpy as np
import cv2

from face_parser.model import extract_masks


dataroot = "../../datasets/multi-pie_crop_patch"
classes = ["train", "test"]
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

"""
for i, id in enumerate(ids):
    print("processing %d ..." % i)
    for light in range(1, 21):
        if "L%s" % light not in CROP_LIGHT_CONDITION:
            shutil.rmtree(os.path.join(dataroot, id, "S001", "L%s" % light))

exit(0)
"""

for cls in classes:
    ids = os.listdir(os.path.join(dataroot, cls))
    ids.sort()
    for idx, id in enumerate(ids):
        print("🔥 processing %s:%s (%d/%d)..." % (cls, id, idx + 1, len(ids)))
        for angle in ANGLES:
            for light in range(20):
                filepath = os.path.join(
                    dataroot, cls, id, RECORDING, angle, "%02d.png" % light
                )
                if not os.path.exists(filepath):
                    continue

                img = Image.open(filepath).convert("RGB")
                masks = extract_masks(img)
                W, H = img.size
                mask_acc = np.zeros((H, W))
                for mask in masks:
                    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
                    dilated = cv2.dilate(mask.squeeze(), kernel, iterations=3)
                    mask_acc += dilated

                mask_acc = (mask_acc > 0)[..., np.newaxis].repeat(3, axis=-1)
                patch = np.array(img) * mask_acc

                result = Image.fromarray(patch)

                new_filepath = os.path.join(
                    dataroot, cls, id, RECORDING, angle, "%02d" % light
                )
                result.save(
                    os.path.join(
                        dataroot,
                        cls,
                        id,
                        RECORDING,
                        angle,
                        "%02d_patch.png" % light,
                    )
                )
