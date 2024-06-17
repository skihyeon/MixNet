import cv2
import numpy as np

import os

def extract_polygons(image_path, recog_result):
    output_dir = './vis/extracted'
    cropped_results = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(recog_result, 'r') as file:
        lines = file.readlines()

    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    for idx, line in enumerate(lines):
        coords = [list(map(int, point.split(','))) for point in line.strip().split()]
        coords = [max(0, coord) for coord in coords[0]]
        
        pts = np.array(coords, np.int32)
        pts = pts.reshape((-1, 1, 2))

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        result = cv2.bitwise_and(image, image, mask=mask)

        x, y, w, h = cv2.boundingRect(pts)
        cropped_result = result[y:y+h, x:x+w]

        if cropped_result.size == 0:
            print(f"Warning: cropped_result is empty for polygon {idx}")
            continue

        output_path = os.path.join(output_dir, f'extracted_polygon_{idx:03d}.png')
        output_path = os.path.abspath(output_path)  # 절대 경로로 변환
        cv2.imwrite(output_path, cropped_result)
        
        cropped_results.append(cropped_result)

    return cropped_results
# cv2.imshow('Extracted Polygon', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

if __name__ == "__main__":
    extract_polygons('./0001.jpg', './0001_infered.txt')