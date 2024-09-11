import unittest
from PIL import Image
import numpy as np
import cv2


def load_and_display_image(image_path):
    image = Image.open(image_path)
    image = np.array(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

def apply_threshold(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def find_and_display_contours(image_rgb, thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def analyze_white_spots(image_rgb):
    white_spots = cv2.inRange(image_rgb, np.array([200, 200, 200]), np.array([255, 255, 255]))
    white_spot_percentage = (np.sum(white_spots > 0) / white_spots.size) * 100
    return white_spot_percentage

def analyze_brown_spots(image_rgb):
    brown_spots = cv2.inRange(image_rgb, np.array([100, 50, 0]), np.array([200, 150, 100]))
    brown_spot_percentage = (np.sum(brown_spots > 0) / brown_spots.size) * 100
    return brown_spot_percentage

class TestImageProcessing(unittest.TestCase):

    def test_load_and_display_image(self):
        image_rgb = load_and_display_image("5.jpg")
        self.assertEqual(image_rgb.shape[2], 3)  # RGB kanalları kontrolü

    def test_apply_threshold(self):
        image_rgb = load_and_display_image("5.jpg")
        thresh = apply_threshold(image_rgb)
        self.assertEqual(len(thresh.shape), 2)  # Grayscale kontrolü

    def test_find_and_display_contours(self):
        image_rgb = load_and_display_image("5.jpg")
        thresh = apply_threshold(image_rgb)
        contours = find_and_display_contours(image_rgb, thresh)
        self.assertGreater(len(contours), 0)  # Kontur varlığı kontrolü

    def test_analyze_white_spots(self):
        image_rgb = load_and_display_image("5.jpg")
        white_spot_percentage = analyze_white_spots(image_rgb)
        self.assertGreaterEqual(white_spot_percentage, 0)  # Leke yüzdesi kontrolü

    def test_analyze_brown_spots(self):
        image_rgb = load_and_display_image("5.jpg")
        brown_spot_percentage = analyze_brown_spots(image_rgb)
        self.assertGreaterEqual(brown_spot_percentage, 0)  # Koyu leke yüzdesi kontrolü


if __name__ == '__main__':
    unittest.main()
