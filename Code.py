import cv2
import numpy as np
from matplotlib import pyplot as plt
from google.colab import drive

drive.mount('/content/drive')

image_path = 'drive/My Drive/download2.jpg'

image = cv2.imread(image_path)

plt.title("Input Image:")

plt.imshow(image)

plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)

plt.title("input Image")

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)

plt.title("Grayscale Image")

plt.imshow(gray, cmap='gray')

plt.show()

if image is not None:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    dilated = cv2.dilate(edges, None, iterations=2)

    eroded = cv2.erode(dilated, None, iterations=1)

    contours, _ = cv2.findContours(eroded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    num_objects = len(contours)

    mask = np.zeros_like(gray)

    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    masked_image = cv2.bitwise_and(image, image, mask=mask)

    masked_gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(masked_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    segmented_contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(image, segmented_contours, -1, (255, 0, 0), 2)

    bounding_boxes = [cv2.boundingRect(contour) for contour in segmented_contours]

    for x, y, w, h in bounding_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    areas = [cv2.contourArea(contour) for contour in segmented_contours]

    plt.figure()

    plt.hist(areas, bins=10, color='cyan')

    plt.xlabel('Area')

    plt.ylabel('Frequency')

    plt.title('Object Area Distribution')

    plt.grid(True)

    plt.show()

    total_area = sum(areas)

    print("Total area of objects:", total_area)

    avg_area = np.mean(areas)

    print("Average area of objects:", avg_area)

    max_area = max(areas)

    max_area_index = areas.index(max_area)

    print(max_area)

    min_area = min(areas)

    print(min_area)

    min_area_index = areas.index(min_area)

    print("List of Areas:", areas)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.title('Object Detection')

    plt.xticks([]), plt.yticks([])

    plt.show()

    rgb_content = np.mean(image, axis=(0, 1))

    print("RGB Content:", rgb_content)
    my_list=[1,'a',2]
    my_list.append(3);
    new_tuple={1,2,3}
