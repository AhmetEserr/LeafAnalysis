


from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

def load_and_display_image(image_path):
    image = Image.open(image_path)
    image = np.array(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image_rgb)
    plt.title('Uploaded Plant Leaf')
    plt.axis('off')
    plt.show()

    return image_rgb

def apply_threshold(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def find_and_display_contours(image_rgb, thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = image_rgb.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

 
    plt.imshow(contour_img)
    plt.title('Leaf Contours')
    plt.axis('off')
    plt.show()

    return contours

def analyze_white_spots(image_rgb):
    white_spots = cv2.inRange(image_rgb, np.array([200, 200, 200]), np.array([255, 255, 255]))
    result = cv2.bitwise_and(image_rgb, image_rgb, mask=white_spots)

    plt.imshow(result)
    plt.title('White Spots on Leaf')
    plt.axis('off')
    plt.show()

    white_spot_percentage = (np.sum(white_spots > 0) / white_spots.size) * 100
    return white_spot_percentage

def analyze_brown_spots(image_rgb):
    brown_spots = cv2.inRange(image_rgb, np.array([100, 50, 0]), np.array([200, 150, 100]))
    result = cv2.bitwise_and(image_rgb, image_rgb, mask=brown_spots)

   
    plt.imshow(result)
    plt.title('Brown Spots on Leaf')
    plt.axis('off')
    plt.show()

    brown_spot_percentage = (np.sum(brown_spots > 0) / brown_spots.size) * 100
    return brown_spot_percentage
def process_image(image_path):
    image_rgb = load_and_display_image(image_path)
    thresh = apply_threshold(image_rgb)
    contours = find_and_display_contours(image_rgb, thresh)
    white_spot_percentage = analyze_white_spots(image_rgb)
    brown_spot_percentage = analyze_brown_spots(image_rgb)

    print(f"Beyaz leke yüzdesi: {white_spot_percentage:.2f}%")
    print(f"Koyu leke yüzdesi: {brown_spot_percentage:.2f}%")


process_image("5.jpg")
process_image("2.jpg")
