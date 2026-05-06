import cv2
import numpy as np
import pytesseract
import re
import glob

# =====================================================================
# INITIAL CONFIGURATION
# =====================================================================
# UNCOMMENT THIS IF YOU ARE ON WINDOWS AND SET YOUR CORRECT PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

IMAGES_FOLDER = 'images/*'

# =====================================================================
# GEOMETRIC FUNCTIONS
# =====================================================================
def order_points(pts):
    """Orders coordinates to consistently represent: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def transform_perspective(image, pts):
    """Applies a perspective transform to flatten the license plate."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0], 
        [maxWidth - 1, 0], 
        [maxWidth - 1, maxHeight - 1], 
        [0, maxHeight - 1]], dtype="float32")
        
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

# =====================================================================
# TEXT PROCESSING FUNCTIONS (SUPPORT FOR SPANISH PLATES)
# =====================================================================
def isolate_and_correct_plate(ocr_text):
    """Cleans the OCR output and validates it against Spanish license plate formats."""
    clean_text = re.sub(r'[^A-Z0-9]', '', ocr_text.upper())
    
    if len(clean_text) < 5:
        return None
        
    # 1. Check NEW Format (0000 XXX)
    new_candidate = clean_text[-7:]
    if len(new_candidate) == 7:
        # Correct common OCR mistakes (e.g., confusing 'O' with '0')
        numbers = new_candidate[:4].replace('O', '0').replace('I', '1').replace('B', '8').replace('S', '5').replace('Z', '2').replace('G', '6')
        letters = new_candidate[4:].replace('0', 'O').replace('1', 'I').replace('8', 'B').replace('5', 'S').replace('2', 'Z')
        new_plate = numbers + letters
        
        if re.match(r"^[0-9]{4}[BCDFGHJKLMNPRSTVWXYZ]{3}$", new_plate):
            return new_plate

    # 2. Check OLD Format (e.g., GE154203, B1234AB)
    # Regex: 1-2 letters + 4-6 numbers + 0-2 letters
    if re.match(r"^[A-Z]{1,2}[0-9]{4,6}[A-Z]{0,2}$", clean_text):
        return clean_text
    
    return None

# =====================================================================
# MAIN PIPELINE
# =====================================================================
def process_image(image_path):
    print(f"\nProcessing: {image_path}")
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        print("Error: Could not load the image.")
        return

    # 1. Resize for PROCESSING (Maintains OCR quality)
    h, w = original_image.shape[:2]
    new_w = 800
    new_h = int(h * (new_w / w))
    image = cv2.resize(original_image, (new_w, new_h))
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered_gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    edges = cv2.Canny(filtered_gray, 30, 200)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_edges = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(closed_edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    
    found_plate = None
    final_text = ""
    plate_contour = None

    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approximation = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        
        if len(approximation) == 4:
            flat_crop = transform_perspective(gray, approximation.reshape(4, 2))
            
            _, binary_crop = cv2.threshold(flat_crop, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            tesseract_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            
            ocr_text = pytesseract.image_to_string(binary_crop, config=tesseract_config)
            
            result = isolate_and_correct_plate(ocr_text)
            
            if result:
                found_plate = flat_crop
                final_text = result
                plate_contour = approximation
                break 

    # 2. Resize ONLY for VISUALIZATION
    if found_plate is not None:
        if len(final_text) == 7 and final_text[:4].isdigit():
            formatted_plate = f"{final_text[:4]} {final_text[4:]}"
        else:
            formatted_plate = final_text 
            
        print(f"License Plate Detected: {formatted_plate}")
        
        cv2.drawContours(image, [plate_contour], -1, (0, 255, 0), 3)
        x, y, box_w, box_h = cv2.boundingRect(plate_contour)
        
        cv2.rectangle(image, (x, y - 40), (x + 220, y), (0,0,0), -1)
        cv2.putText(image, formatted_plate, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # --- SCREEN REDUCTION ---
        # Make a smaller copy of the main image for displaying (e.g., 600px wide)
        visual_width = 600
        screen_scale = visual_width / image.shape[1]
        screen_image = cv2.resize(image, (0, 0), fx=screen_scale, fy=screen_scale)
        
        # Make a smaller copy of the cropped plate (e.g., 250px wide)
        rec_h, rec_w = found_plate.shape[:2]
        crop_scale = 250 / rec_w 
        screen_plate = cv2.resize(found_plate, (int(rec_w * crop_scale), int(rec_h * crop_scale)))
        
        # Show the scaled down copies (OpenCV will auto-size the windows)
        cv2.imshow("Detection", screen_image)
        cv2.imshow("Cropped Plate", screen_plate)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failure: Could not extract the license plate.")

if __name__ == "__main__":
    file_paths = glob.glob(IMAGES_FOLDER)
    for path in file_paths:
        process_image(path)