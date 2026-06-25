import fitz
import os

def pdf_to_images(pdf_path, output_dir="pdf_pages"):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    images = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        image_path = os.path.join(output_dir, f"page_{page_num}.png")
        pix.save(image_path)
        images.append(image_path)

    doc.close()
    return images
