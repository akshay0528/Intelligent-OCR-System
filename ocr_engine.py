import cv2
import easyocr
import pytesseract

_reader = None

def get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'], gpu=True)
    return _reader

def extract_text_easyocr(image_path):
    reader = get_reader()
    image = cv2.imread(image_path)
    if image is None:
        return "", 0.0, []

    results = reader.readtext(image, detail=1, paragraph=False)

    lines = []
    confidences = []
    boxes = []

    for item in results:
        if len(item) >= 3:
            box, text, conf = item[0], item[1], item[2]
            lines.append(str(text))
            confidences.append(float(conf))
            boxes.append(box)

    full_text = "\n".join(lines)
    avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0
    return full_text, avg_conf, boxes

def extract_text_tesseract(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "", 0.0

    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words = []
    confs = []
    for i in range(len(data['text'])):
        word = data['text'][i].strip()
        conf = data['conf'][i]
        if word and conf != '-1':
            words.append(word)
            try:
                confs.append(float(conf))
            except ValueError:
                pass

    full_text = " ".join(words)
    avg_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
    return full_text, avg_conf

def extract_text(image_path):
    boxes = []
    try:
        text, conf, boxes = extract_text_easyocr(image_path)
        if text.strip() and conf > 0.15:
            return text, conf, "EasyOCR", boxes
    except Exception:
        pass

    try:
        text, conf = extract_text_tesseract(image_path)
        return text, conf, "Tesseract", boxes
    except Exception as e:
        return f"OCR Error: {str(e)}", 0.0, "None", boxes

def draw_boxes(image_path, boxes, save_path="boxed.png"):
    image = cv2.imread(image_path)
    if image is None or not boxes:
        return image_path

    for box in boxes:
        pts = [(int(p[0]), int(p[1])) for p in box]
        for i in range(len(pts)):
            cv2.line(image, pts[i], pts[(i + 1) % len(pts)], (0, 255, 0), 2)

    cv2.imwrite(save_path, image)
    return save_path
