# Intelligent OCR System

Document OCR and understanding pipeline using open-source tools only.

## Pipeline
1. Preprocessing — OpenCV: grayscale, denoising, adaptive thresholding, deskew, sharpening.
2. OCR — EasyOCR (GPU), with Tesseract as fallback if confidence is low or EasyOCR fails.
3. LLM correction and extraction — Qwen2.5-3B-Instruct (Hugging Face) corrects OCR errors and extracts structured JSON fields.
4. Classification — keyword-based scoring across document types with a confidence score.
5. Validation — regex rules for email, phone, PAN, Aadhaar, GST, IFSC, date, numeric fields.
6. Document chat — LLM answers questions grounded in the extracted document text.

## Structure
```
app.py
modules/
  preprocessing.py
  ocr_engine.py
  pdf_processor.py
  classifier.py
  llm_processor.py
  validator.py
requirements.txt
```

## Run on Kaggle (GPU notebook)
```
!pip install -r requirements.txt
!pip install pyngrok
!python -m streamlit run app.py & npx localtunnel --port 8501
```

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

Tesseract binary must be installed separately on the OS (`apt install tesseract-ocr` on Linux/Kaggle).
