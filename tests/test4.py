import cv2
import numpy as np

# Load image
image_path = "test1.jpg"  # Change to your image path
image = cv2.imread(image_path)
debug_image = image.copy()
highlight_image = image.copy()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive threshold
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Use morphological operations to detect horizontal lines (row separators)
kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))  # Detect horizontal lines
horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_horizontal)

# Find contours for row detection
contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out small contours and extract row bounding boxes
rows = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 100:  # Ignore small unwanted contours
        rows.append((x, y, w, h))

# Sort rows from top to bottom
rows = sorted(rows, key=lambda x: x[1])

# Draw row bounding boxes (debugging)
for i, (x, y, w, h) in enumerate(rows):
    cv2.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue for row boxes
    cv2.putText(debug_image, f"Row {i+1}", (10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Convert image to HSV for better stamp detection
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define black color range for stamp detection
lower_black = np.array([0, 0, 0], dtype="uint8")
upper_black = np.array([180, 255, 50], dtype="uint8")

# Create stamp mask
stamp_mask = cv2.inRange(hsv, lower_black, upper_black)

# Find stamp contours
stamp_contours, _ = cv2.findContours(stamp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Identify the row containing the stamp
detected_row = None

# Find the rightmost vertical line (signature column boundary)
kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))  # Detect vertical lines
vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_vertical)

# Find contours of vertical lines
vertical_contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get the rightmost vertical line (signature column boundary)
if vertical_contours:
    vertical_contours = sorted(vertical_contours, key=lambda c: cv2.boundingRect(c)[0])
    x_min, _, _, _ = cv2.boundingRect(vertical_contours[-2])  # Left boundary of signature column
    x_max, _, _, _ = cv2.boundingRect(vertical_contours[-1])  # Right boundary

    # Draw signature column boundary (for debugging)
    cv2.line(debug_image, (x_min, 0), (x_min, image.shape[0]), (0, 255, 255), 2)
    cv2.line(debug_image, (x_max, 0), (x_max, image.shape[0]), (0, 255, 255), 2)

# Loop through detected stamp contours
for cnt in stamp_contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # Only check if the stamp is inside the Signature column
    if x_min <= x <= x_max:
        stamp_center_y = y + (h // 2)

        # Draw detected stamp
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(debug_image, "Stamp", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Find which row contains this stamp
        for i, (rx, ry, rw, rh) in enumerate(rows):
            if ry <= stamp_center_y <= ry + rh:
                detected_row = i + 1  # Convert to 1-based index
                
                # Highlight detected row in RED
                cv2.rectangle(highlight_image, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 3)
                cv2.putText(highlight_image, f"Detected Row {detected_row}", (50, ry + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                break

# Save debug image and highlighted row image
cv2.imwrite("debug_image.jpg", debug_image)
cv2.imwrite("highlight_image.jpg", highlight_image)

if detected_row:
    print(f"\n✅ Fixed: Stamp detected in row {detected_row}")
else:
    print("\n❌ Error: Stamp detection failed.")
