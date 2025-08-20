import face_alignment
import cv2
from colorsys import hsv_to_rgb
import math


def show_image(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)


def draw_landmarks(image, landmarks):
    for idx, (x, y) in enumerate(landmarks):
        color = hsv_to_rgb(idx / len(landmarks), 1.0, 1.0)
        color = tuple(map(lambda x: int(x * 255.0), color))
        cv2.circle(image, (int(x), int(y)), 2, color, -1)
        cv2.putText(image, str(idx), (int(x) + 5, int(y)), 0, 0.3, color, 1)

    return image


def get_centers_of_eyes(lm):
    right_eye = ((lm[36][0] + lm[39][0]) // 2, (lm[36][1] + lm[39][1]) // 2)
    left_eye = ((lm[42][0] + lm[45][0]) // 2, (lm[42][1] + lm[45][1]) // 2)
    center_eyes = (
        (right_eye[0] + left_eye[0]) // 2,
        (right_eye[1] + left_eye[1]) // 2,
    )

    return right_eye, left_eye, center_eyes


def align_by_landmarks(image, eyes_lm, mouth_lm):
    height, width = image.shape[:2]
    right_eye, left_eye, center_eyes = eyes_lm
    mouth_eyes_dist = math.sqrt(
        (mouth_lm[0] - center_eyes[0]) ** 2 + (mouth_lm[1] - center_eyes[1]) ** 2
    )
    theta = 0.0

    # rotate the image
    if abs(right_eye[0] - left_eye[0]) < mouth_eyes_dist // 2:  # side-view
        mouth_eyes_dist_y = math.sqrt((mouth_lm[1] - center_eyes[1]) ** 2)
        theta = (math.acos(mouth_eyes_dist_y / mouth_eyes_dist) * 180) / math.pi
        theta = theta * (-1 if center_eyes[0] < mouth_lm[0] else 1)
    else:
        eye_dist_x = math.sqrt((right_eye[0] - left_eye[0]) ** 2)
        eye_dist_y = math.sqrt((right_eye[1] - left_eye[1]) ** 2)
        theta = (math.atan2(eye_dist_y, eye_dist_x) * 180) / math.pi
        theta = theta * (-1 if right_eye[1] > left_eye[1] else 1)

    rotate_mat = cv2.getRotationMatrix2D(center_eyes, theta, 1.0)
    rotated = cv2.warpAffine(
        image,
        rotate_mat,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT,
    )

    # crop the image
    # mouth_dist_x = math.sqrt((mouth_lm[0] - center_eyes[0]) ** 2)
    # theta_plus_phi = (math.asin(mouth_dist_x / mouth_eyes_dist)) * 180 / math.pi
    # phi = abs(theta_plus_phi - theta)
    # new_mouth_eyes_dist_y = int(mouth_eyes_dist * math.cos(math.radians(phi)))

    cropped = rotated[
        int(center_eyes[1] - mouth_eyes_dist) : int(
            center_eyes[1] + mouth_eyes_dist * 2
        ),
        int(center_eyes[0] - mouth_eyes_dist * 1.5) : int(
            center_eyes[0] + mouth_eyes_dist * 1.5
        ),
        :,
    ]

    return cropped


def align_image(image):
    global fa

    preds = fa.get_landmarks(image)

    if preds is None or len(preds) <= 0:
        return None

    preds = preds[0]
    eyes_lm = get_centers_of_eyes(preds)

    # image = draw_landmarks(image, [*eyes_lm, preds[66]])

    aligned = align_by_landmarks(image, eyes_lm, preds[66])

    return aligned
