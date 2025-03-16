import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 🎨 Streamlit UI Styling
st.set_page_config(page_title="Stamp Detection App", layout="centered")

st.title("debug/testing")
st.write("Upload an image containing a table with a stamp, and this app will detect the row where the stamp is present.")

# 📂 Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert to OpenCV format
    image = Image.open(uploaded_file)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR (OpenCV format)

    # Make copies for debugging
    debug_image = image.copy()
    highlight_image = image.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Detect horizontal lines (row separators)
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_horizontal)

    # Find contours for row detection
    contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and extract row bounding boxes
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

    # Convert image to HSV for stamp detection
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

    # Find rightmost vertical lines (signature column boundary)
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_vertical)

    # Find contours of vertical lines
    vertical_contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(vertical_contours) < 2:
        st.error("❌ Not enough vertical lines detected to define the signature column.")
        st.image(debug_image, caption="Debug Image (No signature column detected)", use_column_width=True)
        st.stop()

    # Get the rightmost vertical line (signature column boundary)
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

    # Convert OpenCV images to Streamlit display format
    debug_image = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
    highlight_image = cv2.cvtColor(highlight_image, cv2.COLOR_BGR2RGB)

    # Always show debug image
    st.subheader("🔍 Debug Image")
    st.image(debug_image, caption="Debugging Detection", use_column_width=True)

    # Show detected row image if a stamp is found
    if detected_row:
        st.subheader("✅ Detected Stamp Row")
        st.image(highlight_image, caption="Highlighted Row", use_column_width=True)
        st.success(f"✅ Stamp detected in row **{detected_row}**")
    else:
        st.error("❌ Stamp detection failed.")
