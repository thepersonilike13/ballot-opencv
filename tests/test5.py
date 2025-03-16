import cv2
import numpy as np

# Load image
image_path = "test1.jpg"  # Your ballot image
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to enhance contours
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Debug: Draw all contours
debug_all_contours = image.copy()
cv2.drawContours(debug_all_contours, contours, -1, (0, 255, 255), 2)  # Yellow contours
cv2.imwrite("debug_all_contours.jpg", debug_all_contours)

# Define signature column range
signature_col_min = None
signature_col_max = None

# Detect table structure
lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

horizontal_lines = []
vertical_lines = []

for line in lines:
    x1, y1, x2, y2 = line[0]
    if abs(y2 - y1) < abs(x2 - x1):  # Horizontal lines
        horizontal_lines.append((x1, y1, x2, y2))
    else:  # Vertical lines
        vertical_lines.append((x1, y1, x2, y2))

# Sort vertical lines to get column boundaries
vertical_lines.sort(key=lambda x: x[0])
signature_col_min = vertical_lines[-2][0]  # Left boundary of signature column
signature_col_max = vertical_lines[-1][0]  # Right boundary

# Debug: Draw signature column
debug_signature_col = image.copy()
cv2.rectangle(debug_signature_col, (signature_col_min, 0), (signature_col_max, image.shape[0]), (255, 0, 0), 3)  # Blue box
cv2.imwrite("debug_signature_column.jpg", debug_signature_col)

# Identify the row containing the stamp
detected_row = None
debug_stamp_contours = image.copy()

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    contour_center_x = x + w // 2
    contour_center_y = y + h // 2

    # Ensure it's inside the signature column
    if signature_col_min <= contour_center_x <= signature_col_max:
        cv2.rectangle(debug_stamp_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box around detected stamps

        # Find corresponding row
        for i in range(len(horizontal_lines) - 1):
            row_top = horizontal_lines[i][1]
            row_bottom = horizontal_lines[i + 1][1]

            if row_top <= contour_center_y <= row_bottom:
                detected_row = i + 1
                cv2.rectangle(image, (0, row_top), (image.shape[1], row_bottom), (0, 0, 255), 3)  # Red highlight for correct row
                cv2.putText(image, f"Detected Row {detected_row}", (50, row_top + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                break

# Debug: Save stamp contours detection
cv2.imwrite("debug_stamp_contours.jpg", debug_stamp_contours)

# Save final highlighted image
cv2.imwrite("final_highlighted_row.jpg", image)

# Output result
if detected_row:
    print(f"\n✅ Stamp detected in row {detected_row}")
else:
    print("\n❌ Error: Stamp detection failed.")
