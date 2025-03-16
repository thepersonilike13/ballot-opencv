import cv2
import numpy as np

# Load the image
image_path = "test1.jpg"  # Change this to your correct image path
image = cv2.imread(image_path)
debug_image = image.copy()
highlight_image = image.copy()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection
edges = cv2.Canny(gray, 50, 150)

# Detect lines using Hough Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

# Separate horizontal and vertical lines
horizontal_lines = []
vertical_lines = []

for line in lines:
    x1, y1, x2, y2 = line[0]
    if abs(y2 - y1) < abs(x2 - x1):  # Horizontal line
        horizontal_lines.append((x1, y1, x2, y2))
    else:  # Vertical line
        vertical_lines.append((x1, y1, x2, y2))

# Sort horizontal lines by Y-coordinate (to get rows in order)
horizontal_lines.sort(key=lambda x: x[1])

# Remove duplicate or closely spaced lines (avoid extra grid lines)
filtered_rows = []
row_threshold = 20  # Minimum pixel gap between two rows

for i, (x1, y1, x2, y2) in enumerate(horizontal_lines):
    if i == 0 or (y1 - filtered_rows[-1][1]) > row_threshold:
        filtered_rows.append((x1, y1, x2, y2))

# Sort vertical lines to get column boundaries
vertical_lines.sort(key=lambda x: x[0])

# Define signature column range
x_min = vertical_lines[-2][0]  # Left boundary of signature column
x_max = vertical_lines[-1][0]  # Right boundary

# Convert image to grayscale and apply adaptive thresholding (better for stamp detection)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Find stamp contours
stamp_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Identify row containing the stamp
detected_row = None

for cnt in stamp_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    contour_area = cv2.contourArea(cnt)

    # Ignore small noise and very large objects
    if contour_area < 500 or contour_area > 20000:
        continue

    # Ensure at least 50% of the contour is inside the signature column
    if x + w // 2 >= x_min and x + w // 2 <= x_max:
        stamp_center_y = y + (h // 2)

        # Draw detected stamp on debug image
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(debug_image, "Stamp", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Find which row contains this stamp
        for i in range(len(filtered_rows) - 1):
            y_start = filtered_rows[i][1]
            y_end = filtered_rows[i + 1][1]

            # Draw debugging box in yellow
            cv2.rectangle(debug_image, (0, y_start), (image.shape[1], y_end), (0, 255, 255), 2)

            if y_start <= stamp_center_y <= y_end:
                detected_row = i + 1  # Convert to 1-based index
                
                # Highlight correct detected row in RED on both images
                cv2.rectangle(debug_image, (0, y_start), (image.shape[1], y_end), (0, 0, 255), 3)
                cv2.rectangle(highlight_image, (0, y_start), (image.shape[1], y_end), (0, 0, 255), 3)
                cv2.putText(debug_image, f"Detected Row {detected_row}", (50, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(highlight_image, f"Detected Row {detected_row}", (50, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                break

# Save both debugging and final highlight images
cv2.imwrite("debug_image.jpg", debug_image)
cv2.imwrite("highlight_image.jpg", highlight_image)

if detected_row:
    print(f"\n✅ Fixed: Stamp detected in row {detected_row}")
else:
    print("\n❌ Error: Stamp detection failed.")
