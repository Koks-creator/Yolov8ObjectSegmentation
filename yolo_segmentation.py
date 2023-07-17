from dataclasses import dataclass
from typing import Tuple, List, Union
from ultralytics import YOLO
import cv2
import numpy as np

np.random.seed(20)


@dataclass()
class YOLOSegmentation:
    model_path: str
    contour_color: Tuple[int, int, int] = (255, 255, 255)
    font_size: int = 3
    font_thickness: int = 2

    def __post_init__(self) -> None:
        self.model = YOLO(self.model_path)
        with open("classes.txt") as f:
            self.classes = f.read().splitlines()
            self.color_list = np.random.randint(low=0, high=255, size=(len(self.classes), 3))

    def detect(self, img: np.array) -> Tuple[np.array, np.array, List[np.array], np.array]:
        segmentation_contours_idx = []
        height, width, channels = img.shape

        results = self.model.predict(source=img.copy(), save=False, save_txt=False)
        result = results[0]

        for seg in result.masks.segments:
            seg[:, 0] *= width
            seg[:, 1] *= height

            segment = np.array(seg, dtype=np.int32)
            segmentation_contours_idx.append(segment)

        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
        scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)

        return bboxes, class_ids, segmentation_contours_idx, scores

    @staticmethod
    def draw_mask(img: np.array, mask: np.array, color: List[Union[int, int, int]], alpha=0.5) -> np.array:
        overlay = img.copy()

        cv2.drawContours(img, [mask], -1, color, -1)
        final_img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        return final_img

    def draw_detections(self, img: np.array, draw_bbox: bool = True, draw_contour: bool = True,
                        draw_mask: bool = True, labels: bool = True) -> Tuple[np.array, np.array, list, list]:
        drawing_img = img.copy()
        mask = np.zeros_like(img)
        cropped_images_img = None

        bboxes, classess, segmentations, scores = self.detect(img)
        for bbox, class_id, seg, score in zip(bboxes, classess, segmentations, scores):
            x1, y1, x2, y2 = bbox

            class_color = [int(c) for c in self.color_list[class_id]]
            if draw_mask:
                drawing_img = self.draw_mask(drawing_img, seg, class_color)
            if draw_contour:
                cv2.drawContours(drawing_img, [seg], -1, self.contour_color, 2)
            if draw_bbox:
                cv2.rectangle(drawing_img, (x1, y1), (x2, y2), class_color, 3)
            if labels:
                cv2.putText(drawing_img, f"{self.classes[class_id]}", (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN,
                            self.font_size, class_color, self.font_thickness)

            # Cropping objects
            cv2.drawContours(mask, [seg], -1, (255, 255, 255), -1)
            cropped_images_img = np.zeros_like(img)
            cropped_images_img[mask == 255] = img[mask == 255]

        return drawing_img, cropped_images_img, segmentations, bboxes


if __name__ == '__main__':
    IMAGES_FOLDER = "images"
    yolo = YOLOSegmentation("yolov8m-seg.pt")

    img = cv2.imread(fr"{IMAGES_FOLDER}\d.png")
    background_image = cv2.imread(fr"{IMAGES_FOLDER}\house.jpg")

    img = cv2.resize(img, None, fx=0.7, fy=0.7)
    background_image = cv2.resize(background_image, (img.shape[1], img.shape[0]))
    print(background_image.shape)

    drawing_img, cropped_objects_img, segmentations, bboxes = yolo.draw_detections(img)

    # plain_img = np.zeros_like(img)
    # plain_img[:, :, :] = 0, 0, 100
    offset = (300, 100)

    for seg, bbox in zip(segmentations, bboxes):
        cv2.drawContours(background_image, [seg], -1, (0, 0, 0), -1)  # comment it when moving object
        # cv2.drawContours(background_image, [seg], -1, (0, 0, 0), -1)

        # move object
        # M = cv2.moments(seg)
        # if M['m00'] != 0:
        #     cx = int(M['m10'] / M['m00'])
        #     cy = int(M['m01'] / M['m00'])
        #
        # x1, y1, x2, y2 = bbox
        # cropped_object = cropped_objects_img[y1:y2, x1:x2]
        # h, w, _ = cropped_object.shape
        # cropped_objects_img = np.zeros_like(img)
        # cv2.drawContours(background_image, [seg], -1, (0, 0, 0), -1, offset=offset)
        # cv2.drawContours(cropped_objects_img, [seg], -1, (0, 0, 0), -1, offset=offset)
        #
        # cropped_objects_img[y1+offset[1]:y1+offset[1]+h, x1+offset[0]:x1+offset[0]+w] = cropped_object
        #
        # cv2.circle(cropped_objects_img, (x1+offset[0], y1+offset[1]), 7, (0, 0, 255), -1)
        # cv2.circle(cropped_objects_img, (x1+offset[0]+w, y1+offset[1]+h), 7, (0, 0, 255), -1)

    res = cv2.add(background_image, cropped_objects_img)
    # res = cv2.add(plain_img, cropped_objects_img)

    cv2.imshow("res", res)
    cv2.imshow("cropped_objects_img", cropped_objects_img)
    cv2.imshow("background_image", background_image)

    cv2.imshow("drawing_img", drawing_img)
    cv2.waitKey(0)
