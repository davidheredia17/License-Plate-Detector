# Automatic License Plate Recognition (ALPR) System

## Overview
This project is a Python-based computer vision application designed to automatically detect, extract, and read vehicle license plates from static images. It leverages OpenCV for image processing and geometric transformations, and Google's Tesseract engine for Optical Character Recognition (OCR). The pipeline is specifically tuned to handle perspective distortions and validate standard Spanish license plate formats.

## Key Features
* **Contour Detection:** Utilizes Bilateral Filtering and Canny edge detection to isolate rectangular bounding boxes within complex environments.
* **Geometrical Perspective Correction:** Implements a custom mathematical algorithm to transform tilted or skewed license plates into a flat, top-down perspective, significantly improving OCR accuracy.
* **Optical Character Recognition:** Processes binary-thresholded plate crops through Tesseract OCR.
* **Format Validation:** Employs Regular Expressions (Regex) to clean OCR noise and validate outputs against traditional and modern Spanish license plate syntax.
* **Resolution Management:** Decouples the processing resolution (high-res for accurate OCR) from the visualization resolution (scaled-down for UI presentation), preventing distorted or oversized display windows.

## Technical Stack
* **Python 3.x**
* **OpenCV** (`opencv-python`): Image manipulation, morphological operations, and contour logic.
* **NumPy**: Matrix operations and point coordinate management.
* **PyTesseract**: Text extraction engine.
* **Regex** (`re`): Data sanitization.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/License-Plate-Detector.git
   cd License-Plate-Detector
   ```

2. **Install Python dependencies:**
   It is highly recommended to use a virtual environment (`venv`).
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR:**
   * **Windows:** Download the binary from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki). Ensure the path in `main.py` points to your installation directory (e.g., `C:\Program Files\Tesseract-OCR\tesseract.exe`).
   * **Linux/macOS:** Run `sudo apt install tesseract-ocr` or `brew install tesseract`.

## Usage

Place the target images in the `images/` directory and execute the main script:

```bash
python main.py
```

The system will iterate through the images, output the parsed license plate to the console, and display the localized bounding box along with the perspective-corrected crop. Press any key to advance to the next image.

## Results

Here is an example of the algorithm successfully isolating and reading a license plate, even with challenging lighting and perspective:

![Detection Process](assets/detected_1248BFX.jpg)
![Detection Process](assets/detected_GE157844.jpg)

*The algorithm flattens the plate and applies binary thresholding before sending it to Tesseract.*

## Development Notes
*Development of this project was assisted by AI coding tools for algorithm refinement and code structuring.*