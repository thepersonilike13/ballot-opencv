import cv2
import numpy as np

# Load the image
image_path = "test1.jpg"
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours to visualize detected edges
contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Convert to HSV to detect dark regions (possible stamp)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define a threshold to detect dark regions (adjust as needed)
lower_black = np.array([0, 0, 0], dtype="uint8")
upper_black = np.array([180, 255, 50], dtype="uint8")

# Create a mask
stamp_mask = cv2.inRange(hsv, lower_black, upper_black)

# Find contours for the stamp
stamp_contours, _ = cv2.findContours(stamp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw detected stamps
for cnt in stamp_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Save the output
cv2.imwrite("detected_stamp.jpg", image)
cv2.imwrite("detected_contours.jpg", contour_image)

# Show result (for local execution)
# cv2.imshow("Detected Stamp", image)
# cv2.imshow("Contours", contour_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
