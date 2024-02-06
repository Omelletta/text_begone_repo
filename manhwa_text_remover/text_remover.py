import cv2
import pytesseract


def remove_text(p_img, img):
    buff = 0.0
    # Configure Tesseract to recognize Korean text
    custom_config = r'--oem 3 --psm 6 -l kor'
    # Detect text in the image
    d = pytesseract.image_to_data(p_img, config=custom_config, output_type=pytesseract.Output.DICT)
    
    n_boxes = len(d['text'])
    print(d['conf'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 80:  # Confidence threshold to filter out weak detections
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            # Draw over img
            wbuff = round(w*buff)
            hbuff = round(h*buff)
            img = cv2.rectangle(img, (x-wbuff, y-hbuff), (x + w+wbuff, y + h+hbuff), (0, 0, 0), -1) # Replace -1 with thickness for a border

    return img
