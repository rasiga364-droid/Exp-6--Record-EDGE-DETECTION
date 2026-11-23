# Exp-6--Record-EDGE-DETECTION

# Aim
To perform edge detection using Sobel, Roberts, Prewitt, Laplacian, and Canny edge detectors.

# Software Required
 Anaconda – Python 3.7
 Jupyter Notebook / VS Code
 OpenCV (cv2)
 NumPy
 Matplotlib
# Algorithm
# Step 1:
Import all the necessary modules for the program.

# Step 2:
Load an image using cv2.imread().

# Step 3:
Convert the image to grayscale.

# Step 4:
Apply Sobel operator using OpenCV to detect edges.

# Step 5:
Apply Prewitt operator using custom kernels.

# Step 6:
Apply Roberts operator using custom kernels.

# Step 7:
Apply Laplacian operator using OpenCV.

# Step 8:
Apply Canny edge detector using OpenCV.

# Step 9:
Display all edge-detected images for comparison.

Developed By
Name: Rasiga s
Register No: 212225220081
# ORIGINAL IMAGE

```
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("BMW4.jpg")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')
plt.show()

```
# output

S
<img width="228" height="142" alt="image" src="https://github.com/user-attachments/assets/8efe60dd-65d0-420b-9149-e04d7bd85865" />


# Sobel Edge Detector
```
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)  
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)  
sobel_combined = cv2.magnitude(sobel_x, sobel_y)  
plt.imshow(sobel_combined, cmap='gray')
plt.title('Sobel Edge Detection')
plt.axis('off')
```
# OUTPUT
<img width="225" height="150" alt="image" src="https://github.com/user-attachments/assets/beab2b3f-6065-4c7a-86e1-374939375aa5" />

# Prewitt Edge Detector

```
image = cv2.imread("BMW4.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

prewitt_x = np.array([[1, 0, -1],
                      [1, 0, -1],
                      [1, 0, -1]])

prewitt_y = np.array([[1, 1, 1],
                      [0, 0, 0],
                      [-1, -1, -1]])

prewitt_x_edge = cv2.filter2D(gray, cv2.CV_32F, prewitt_x)
prewitt_y_edge = cv2.filter2D(gray, cv2.CV_32F, prewitt_y)

prewitt = cv2.magnitude(prewitt_x_edge, prewitt_y_edge)

plt.imshow(prewitt, cmap='gray')
plt.title('Prewitt Edge Detection')
plt.axis('off')
plt.show()
```
# OUTPUT
<img width="225" height="146" alt="image" src="https://github.com/user-attachments/assets/d027e784-af47-4362-a729-8db239ffba54" />

# Roberts Edge Detector
```
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(BMW4.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

roberts_x = np.array([[1, 0],
                      [0, -1]], dtype=np.float32)

roberts_y = np.array([[0, 1],
                      [-1, 0]], dtype=np.float32)

gx = cv2.filter2D(gray, cv2.CV_32F, roberts_x)
gy = cv2.filter2D(gray, cv2.CV_32F, roberts_y)

roberts = cv2.magnitude(gx, gy)
roberts = cv2.normalize(roberts, None, 0, 255, cv2.NORM_MINMAX)
roberts = roberts.astype(np.uint8)

plt.imshow(roberts, cmap="gray")
plt.title("Roberts Edge Detection")
plt.axis("off")
plt.show()
```
# OUTPUT
<img width="220" height="142" alt="image" src="https://github.com/user-attachments/assets/72ca5f0f-17ec-45ed-89e6-0c81955f7c3d" />

# Laplacian Edge Detector
```
import cv2
import matplotlib.pyplot as plt
laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
laplacian_8bit = cv2.convertScaleAbs(laplacian)
plt.imshow(laplacian_8bit, cmap='gray')
plt.title('Laplacian Edge Detection')
plt.axis('off')
plt.show()
```
# OUTPUT
<img width="223" height="150" alt="image" src="https://github.com/user-attachments/assets/525d92b5-fff4-4120-9172-af892a296da3" />


# Canny Edge Detector

```
import cv2
import matplotlib.pyplot as plt
canny_edges = cv2.Canny(gray_image, 50, 150)
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')
plt.show()
```
# OUTPUT
<img width="230" height="155" alt="image" src="https://github.com/user-attachments/assets/62d2a650-dc27-4915-8cbb-580016a1345b" />

# Result
Thus, edges are successfully detected using Sobel, Prewitt, Roberts, Laplacian, and Canny edge detection techniques. Each method highlights edges differently based on gradient and intensity variations, improving feature extraction and analysis.
