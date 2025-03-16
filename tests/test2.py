import cv2
import numpy as np

# Load the image
image_path = "test1.jpg"  # Change this to your correct image path
image = cv2.imread(image_path)

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

# Debugging: Draw valid row lines
for i, (x1, y1, x2, y2) in enumerate(filtered_rows):
    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue lines for row boundaries
    cv2.putText(image, f"Row {i+1}", (10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Define signature column range
x_min = vertical_lines[-2][0]  # Left boundary of signature column
x_max = vertical_lines[-1][0]  # Right boundary

# Convert image to HSV for better detection
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define black color range (stamp detection)
lower_black = np.array([0, 0, 0], dtype="uint8")
upper_black = np.array([180, 255, 50], dtype="uint8")

# Create mask
stamp_mask = cv2.inRange(hsv, lower_black, upper_black)

# Find stamp contours
stamp_contours, _ = cv2.findContours(stamp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Identify row containing the stamp
detected_row = None

for cnt in stamp_contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # Only check if the stamp is inside the Signature column
    if x_min <= x <= x_max:
        stamp_center_y = y + (h // 2)

        # Draw detected stamp
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(image, "Stamp", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Find which row contains this stamp
        for i in range(len(filtered_rows) - 1):
            y_start = filtered_rows[i][1]
            y_end = filtered_rows[i + 1][1]

            # Draw debugging box in yellow
            cv2.rectangle(image, (0, y_start), (image.shape[1], y_end), (0, 255, 255), 2)

            if y_start <= stamp_center_y <= y_end:
                detected_row = i + 1  # Convert to 1-based index
                
                # Highlight correct detected row in RED
                cv2.rectangle(image, (0, y_start), (image.shape[1], y_end), (0, 0, 255), 3)
                cv2.putText(image, f"Detected Row {detected_row}", (50, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                break

# Save the new debugging output
cv2.imwrite("debugged_fixed_stamp_detection.jpg", image)

if detected_row:
    print(f"\n✅ Fixed: Stamp detected in row {detected_row}")
else:
    print("\n❌ Error: Stamp detection failed.")
