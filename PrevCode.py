# from PIL import Image
# import matplotlib.pyplot as plt

# # Load the image
# image_path = "1.jpg"
# image = Image.open(image_path)

# # Display the image
# plt.imshow(image)
# plt.axis('off')
# plt.show()


# import cv2
# import numpy as np

# # Load the image using OpenCV
# image_cv = cv2.imread(image_path)

# # Convert the image to grayscale
# gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

# # Apply a threshold to segment the leaf
# _, thresholded_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# # Display the grayscale and thresholded images
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# ax[0].imshow(gray_image, cmap='gray')
# ax[0].set_title('Grayscale Image')
# ax[0].axis('off')

# ax[1].imshow(thresholded_image, cmap='gray')
# ax[1].set_title('Thresholded Image')
# ax[1].axis('off')

# plt.show()


# # Convert the image to the HSV color space for better color segmentation
# hsv_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)

# # Define color range for healthy (green) and diseased (red) leaves
# green_lower = np.array([35, 40, 40])
# green_upper = np.array([85, 255, 255])

# red_lower = np.array([0, 50, 50])
# red_upper = np.array([10, 255, 255])
# red_lower2 = np.array([170, 50, 50])
# red_upper2 = np.array([180, 255, 255])

# # Create masks for green and red areas
# green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
# red_mask1 = cv2.inRange(hsv_image, red_lower, red_upper)
# red_mask2 = cv2.inRange(hsv_image, red_lower2, red_upper2)
# red_mask = cv2.bitwise_or(red_mask1, red_mask2)

# # Calculate the areas
# green_area = np.sum(green_mask > 0)
# red_area = np.sum(red_mask > 0)

# # Display the original image with masks
# fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# ax[0].imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
# ax[0].set_title('Original Image')
# ax[0].axis('off')

# ax[1].imshow(green_mask, cmap='gray')
# ax[1].set_title('Green Areas (Healthy)')
# ax[1].axis('off')

# ax[2].imshow(red_mask, cmap='gray')
# ax[2].set_title('Red Areas (Diseased)')
# ax[2].axis('off')

# plt.show()

# green_area, red_area

# def suggest_vitamins_and_elements(disease_type):
#     suggestions = {
#         "general_disease": {
#             "Azot (N)": "Bitki büyümesi ve yeşil yapraklar için gereklidir. Amonyum nitrat veya üre gübresi kullanabilirsiniz.",
#             "Fosfor (P)": "Kök gelişimi ve çiçeklenme için önemlidir. Süper fosfat veya kemik unu kullanabilirsiniz.",
#             "Potasyum (K)": "Genel bitki sağlığı, hastalık direnci ve meyve kalitesi için gereklidir. Potasyum sülfat veya potasyum klorür kullanabilirsiniz.",
#             "Kalsiyum (Ca)": "Hücre duvarlarının sağlamlığı ve kök gelişimi için gereklidir. Kalsiyum karbonat veya kalsiyum nitrat kullanabilirsiniz.",
#             "Magnezyum (Mg)": "Klorofil üretimi için gereklidir ve fotosentezde önemli bir rol oynar. Magnezyum sülfat (Epsom tuzu) kullanabilirsiniz.",
#             "Demir (Fe)": "Klorofil sentezi için gereklidir. Demir sülfat veya şelatlı demir kullanabilirsiniz.",
#             "Çinko (Zn)": "Enzim fonksiyonları ve bitki büyümesi için gereklidir. Çinko sülfat kullanabilirsiniz."
#         }
#     }

#     return suggestions.get(disease_type, "Belirtilen hastalık türü için öneri bulunamadı.")

# # Hastalıklı yaprak için önerileri alalım
# disease_type = "general_disease"
# recommendations = suggest_vitamins_and_elements(disease_type)

# for element, recommendation in recommendations.items():
#     print(f"{element}: {recommendation}")


# def suggest_vitamins_and_elements(disease_type):
#     suggestions = {
#         "general_disease": {
#             "Azot (N)": "Bitki büyümesi ve yeşil yapraklar için gereklidir. Amonyum nitrat veya üre gübresi kullanabilirsiniz.",
#             "Fosfor (P)": "Kök gelişimi ve çiçeklenme için önemlidir. Süper fosfat veya kemik unu kullanabilirsiniz.",
#             "Potasyum (K)": "Genel bitki sağlığı, hastalık direnci ve meyve kalitesi için gereklidir. Potasyum sülfat veya potasyum klorür kullanabilirsiniz.",
#             "Kalsiyum (Ca)": "Hücre duvarlarının sağlamlığı ve kök gelişimi için gereklidir. Kalsiyum karbonat veya kalsiyum nitrat kullanabilirsiniz.",
#             "Magnezyum (Mg)": "Klorofil üretimi için gereklidir ve fotosentezde önemli bir rol oynar. Magnezyum sülfat (Epsom tuzu) kullanabilirsiniz.",
#             "Demir (Fe)": "Klorofil sentezi için gereklidir. Demir sülfat veya şelatlı demir kullanabilirsiniz.",
#             "Çinko (Zn)": "Enzim fonksiyonları ve bitki büyümesi için gereklidir. Çinko sülfat kullanabilirsiniz."
#         }
#     }

#     return suggestions.get(disease_type, "Belirtilen hastalık türü için öneri bulunamadı.")

# # Hastalıklı yaprak için önerileri alalım
# disease_type = "general_disease"
# recommendations = suggest_vitamins_and_elements(disease_type)

# print("Yaprak hastalığını iyileştirmek için önerilen vitaminler ve elementler:")
# for element, recommendation in recommendations.items():
#     print(f"{element}: {recommendation}")

# print("\nYaprak hastalığını iyileştirmek için uygulama adımları:")
# print("1. Bitkinin kök bölgesine yukarıda belirtilen besinleri ve mineralleri uygulayın.")
# print("2. Toprağın pH seviyesini kontrol edin ve gerekirse ayarlayın.")
# print("3. Bitkiyi düzenli olarak sulayın, ancak aşırı sulamaktan kaçının.")
# print("4. Hastalıklı yaprakları temizleyin ve bitkiden uzaklaştırın.")
# print("5. Bitkinin yeterli güneş ışığı aldığından emin olun.")

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# # Resmi yükle
# image_path = "1.jpg"
# image = cv2.imread(image_path)

# # Resmi HSV renk uzayına dönüştür
# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# # Sağlıklı (yeşil) yaprak bölgeleri için renk aralığı
# green_lower = np.array([35, 40, 40])
# green_upper = np.array([85, 255, 255])

# # Hastalıklı (kırmızı) yaprak bölgeleri için renk aralığı
# red_lower1 = np.array([0, 50, 50])
# red_upper1 = np.array([10, 255, 255])
# red_lower2 = np.array([170, 50, 50])
# red_upper2 = np.array([180, 255, 255])

# # Maskeleri oluştur
# green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
# red_mask1 = cv2.inRange(hsv_image, red_lower1, red_upper1)
# red_mask2 = cv2.inRange(hsv_image, red_lower2, red_upper2)
# red_mask = cv2.bitwise_or(red_mask1, red_mask2)

# # Alan hesaplamaları
# green_area = np.sum(green_mask > 0)
# red_area = np.sum(red_mask > 0)

# # Orijinal görüntü ve maskeleri göster
# fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# ax[0].set_title('Orijinal Resim')
# ax[0].axis('off')

# ax[1].imshow(green_mask, cmap='gray')
# ax[1].set_title('Yeşil Bölgeler (Sağlıklı)')
# ax[1].axis('off')

# ax[2].imshow(red_mask, cmap='gray')
# ax[2].set_title('Kırmızı Bölgeler (Hastalıklı)')
# ax[2].axis('off')

# plt.show()

# green_area, red_area



# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# # Load the image
# image_path = '1.jpg'
# image = cv2.imread(image_path)

# # Convert to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Apply Gaussian Blur
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# # Apply Thresholding
# _, thresholded = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# # Find contours
# contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Draw contours on the original image
# result_image = image.copy()
# cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)

# # Analyze the spots
# spot_count = len(contours)
# spot_areas = [cv2.contourArea(contour) for contour in contours]

# # Display results
# print(f'Number of spots detected: {spot_count}')
# print(f'Areas of detected spots: {spot_areas}')

# # Show images
# plt.figure(figsize=(10, 10))

# plt.subplot(1, 3, 1)
# plt.title('Original Image')
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# plt.subplot(1, 3, 2)
# plt.title('Thresholded Image')
# plt.imshow(thresholded, cmap='gray')

# plt.subplot(1, 3, 3)
# plt.title('Detected Spots')
# plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))

# plt.show()



# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# # Görüntüyü yükle
# image_path = '2.jpg'
# image = cv2.imread(image_path)

# # Gri tonlamaya çevir
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Gaussian Blur uygula
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# # Eşikleme uygula
# _, thresholded = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# # Konturları bul
# contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Orijinal görüntü üzerinde konturları çiz
# result_image = image.copy()
# cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)

# # Noktaları analiz et
# spot_count = len(contours)
# spot_areas = [cv2.contourArea(contour) for contour in contours]

# # Ortalama alan büyüklüğü
# average_spot_area = np.mean(spot_areas) if spot_areas else 0

# # Renk analizi için HSV renk uzayına çevir
# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv_image, (36, 25, 25), (86, 255,255))  # Yeşil renk için maske
# masked_image = cv2.bitwise_and(image, image, mask=mask)

# # Renk özellikleri
# mean_val = cv2.mean(image, mask=mask)

# # Sonuçları göster
# print(f'Tespit edilen nokta sayısı: {spot_count}')
# print(f'Tespit edilen noktaların alanları: {spot_areas}')
# print(f'Ortalama nokta alanı: {average_spot_area}')
# print(f'Yeşil renk ortalama değeri (HSV): {mean_val}')

# # Görüntüleri göster
# plt.figure(figsize=(10, 10))

# plt.subplot(1, 3, 1)
# plt.title('Orijinal Görüntü')
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# plt.subplot(1, 3, 2)
# plt.title('Eşiklenmiş Görüntü')
# plt.imshow(thresholded, cmap='gray')

# plt.subplot(1, 3, 3)
# plt.title('Tespit Edilen Noktalar')
# plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))

# plt.show()


# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# # Görüntüyü yükle
# image_path = '5.jpg'
# image = cv2.imread(image_path)

# # Gri tonlamaya çevir
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Gaussian Blur uygula
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# # Eşikleme uygula
# _, thresholded = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# # Konturları bul
# contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Orijinal görüntü üzerinde konturları çiz
# result_image = image.copy()
# cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)

# # Noktaları analiz et
# spot_count = len(contours)
# spot_areas = [cv2.contourArea(contour) for contour in contours]

# # Ortalama alan büyüklüğü
# average_spot_area = np.mean(spot_areas) if spot_areas else 0

# # Renk analizi için HSV renk uzayına çevir
# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# mask = cv2.inRange(hsv_image, (36, 25, 25), (86, 255,255))  # Yeşil renk için maske
# masked_image = cv2.bitwise_and(image, image, mask=mask)

# # Renk özellikleri
# mean_val = cv2.mean(image, mask=mask)

# # Sonuçları göster
# print(f'Tespit edilen nokta sayısı: {spot_count}')
# print(f'Tespit edilen noktaların alanları: {spot_areas}')
# print(f'Ortalama nokta alanı: {average_spot_area}')
# print(f'Yeşil renk ortalama değeri (HSV): {mean_val}')

# # Görüntüleri göster
# plt.figure(figsize=(10, 10))

# plt.subplot(1, 3, 1)
# plt.title('Orijinal Görüntü')
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# plt.subplot(1, 3, 2)
# plt.title('Eşiklenmiş Görüntü')
# plt.imshow(thresholded, cmap='gray')

# plt.subplot(1, 3, 3)
# plt.title('Tespit Edilen Noktalar')
# plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))

# plt.show()



# from PIL import Image
# import matplotlib.pyplot as plt
# import cv2
# import numpy as np

# # Load the image
# image_path = '5.jpg'
# image = Image.open(image_path)
# image = np.array(image)

# # Convert the image to RGB
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Display the image
# plt.imshow(image_rgb)
# plt.title('Uploaded Plant Leaf')
# plt.axis('off')
# plt.show()

# # Convert to grayscale
# gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

# # Apply thresholding to segment the leaf
# _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# # Find contours of the leaf
# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Draw contours on the original image
# contour_img = image_rgb.copy()
# cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

# # Display the image with contours
# plt.imshow(contour_img)
# plt.title('Leaf Contours')
# plt.axis('off')
# plt.show()

# # Analyze the leaf for white powdery spots
# white_spots = cv2.inRange(image_rgb, np.array([200, 200, 200]), np.array([255, 255, 255]))
# result = cv2.bitwise_and(image_rgb, image_rgb, mask=white_spots)

# # Display the white spots
# plt.imshow(result)
# plt.title('White Spots on Leaf')
# plt.axis('off')
# plt.show()

# # Check if there are significant white spots indicating powdery mildew
# white_spot_percentage = (np.sum(white_spots > 0) / white_spots.size) * 100

# white_spot_percentage



# # Load the new image
# image_path_2 = '2.jpg'
# image_2 = Image.open(image_path_2)
# image_2 = np.array(image_2)

# # Convert the image to RGB
# image_rgb_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)

# # Display the image
# plt.imshow(image_rgb_2)
# plt.title('Uploaded Plant Leaf')
# plt.axis('off')
# plt.show()

# # Convert to grayscale
# gray_2 = cv2.cvtColor(image_rgb_2, cv2.COLOR_RGB2GRAY)

# # Apply thresholding to segment the leaf
# _, thresh_2 = cv2.threshold(gray_2, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# # Find contours of the leaf
# contours_2, _ = cv2.findContours(thresh_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Draw contours on the original image
# contour_img_2 = image_rgb_2.copy()
# cv2.drawContours(contour_img_2, contours_2, -1, (0, 255, 0), 2)

# # Display the image with contours
# plt.imshow(contour_img_2)
# plt.title('Leaf Contours')
# plt.axis('off')
# plt.show()

# # Analyze the leaf for brown spots
# brown_spots = cv2.inRange(image_rgb_2, np.array([100, 50, 0]), np.array([200, 150, 100]))
# result_2 = cv2.bitwise_and(image_rgb_2, image_rgb_2, mask=brown_spots)

# # Display the brown spots
# plt.imshow(result_2)
# plt.title('Brown Spots on Leaf')
# plt.axis('off')
# plt.show()

# # Check if there are significant brown spots indicating a potential disease
# brown_spot_percentage = (np.sum(brown_spots > 0) / brown_spots.size) * 100

# brown_spot_percentage