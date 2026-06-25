import streamlit as st
import os
import json
import pandas as pd

from modules.preprocessing import preprocess_image
from modules.ocr_engine import extract_text, draw_boxes
from modules.pdf_processor import pdf_to_images
from modules.classifier import classify_document
from modules.llm_processor import correct_and_extract, answer_question
from modules.validator import validate_fields

st.set_page_config(page_title="Intelligent OCR System", layout="wide", page_icon="📄")

st.markdown("""
<style>
.stApp { background-color: #0E1117; color: #FAFAFA; }
.metric-box { background-color: #1A1D23; padding: 15px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("📄 Intelligent OCR System")
st.caption("Open-source OCR + LLM document understanding pipeline")

if "history" not in st.session_state:
    st.session_state.history = []
if "processed_file_id" not in st.session_state:
    st.session_state.processed_file_id = None
if "results" not in st.session_state:
    st.session_state.results = None

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["jpg", "jpeg", "png", "pdf"]
)

if uploaded_file:
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state.processed_file_id != file_id:
        os.makedirs("temp", exist_ok=True)
        input_path = os.path.join("temp", uploaded_file.name)
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file.name.lower().endswith(".pdf"):
            pages = pdf_to_images(input_path)
            image_path = pages[0]
            if len(pages) > 1:
                st.info(f"PDF has {len(pages)} pages. Processing page 1.")
        else:
            image_path = input_path

        processed_path = preprocess_image(image_path, save_path="temp/processed.png")

        with st.spinner("Running OCR..."):
            raw_text, ocr_confidence, engine_used, boxes = extract_text(image_path)

        doc_type, classification_confidence = classify_document(raw_text)

        with st.spinner("Correcting text and extracting fields with LLM (first run loads the model, can take a minute)..."):
            corrected_text, extracted_fields = correct_and_extract(raw_text, doc_type)

        validation_results = validate_fields(extracted_fields)

        boxed_path = draw_boxes(image_path, boxes, save_path="temp/boxed.png") if boxes else None

        st.session_state.results = {
            "image_path": image_path,
            "processed_path": processed_path,
            "boxed_path": boxed_path,
            "raw_text": raw_text,
            "ocr_confidence": ocr_confidence,
            "engine_used": engine_used,
            "doc_type": doc_type,
            "classification_confidence": classification_confidence,
            "corrected_text": corrected_text,
            "extracted_fields": extracted_fields,
            "validation_results": validation_results,
        }
        st.session_state.processed_file_id = file_id
        st.session_state.history = []

    r = st.session_state.results
    image_path = r["image_path"]
    processed_path = r["processed_path"]
    boxed_path = r["boxed_path"]
    raw_text = r["raw_text"]
    ocr_confidence = r["ocr_confidence"]
    engine_used = r["engine_used"]
    doc_type = r["doc_type"]
    classification_confidence = r["classification_confidence"]
    corrected_text = r["corrected_text"]
    extracted_fields = r["extracted_fields"]
    validation_results = r["validation_results"]

    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 OCR & Image", "🧠 Classification & Fields", "✅ Validation", "💬 Chat"
    ])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image_path, use_container_width=True)
        with col2:
            st.subheader("Processed Image")
            st.image(processed_path, use_container_width=True)

        st.metric("OCR Engine Used", engine_used)
        st.metric("OCR Confidence", f"{ocr_confidence*100:.1f}%")

        st.subheader("Raw OCR Text")
        st.code(raw_text)

        st.subheader("LLM-Corrected Text")
        st.code(corrected_text)

        if boxed_path:
            st.subheader("Bounding Boxes")
            st.image(boxed_path, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Document Type", doc_type)
        with col2:
            st.metric("Classification Confidence", f"{classification_confidence}%")

        st.subheader("Extracted Structured Fields (JSON)")
        st.json(extracted_fields)

        json_str = json.dumps(extracted_fields, indent=2)
        st.download_button("📥 Download JSON", json_str, "extracted_data.json", "application/json")

        try:
            df = pd.DataFrame(list(extracted_fields.items()), columns=["Field", "Value"])
            csv = df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "extracted_data.csv", "text/csv")
        except Exception:
            pass

    with tab3:
        st.subheader("Validation Results")
        if validation_results:
            for field, result in validation_results.items():
                status = result["valid"]
                if status is True:
                    st.success(f"{field}: {result['value']} — Valid")
                elif status is False:
                    st.error(f"{field}: {result['value']} — Invalid format")
                else:
                    st.info(f"{field}: {result['value']} — No validator defined")
        else:
            st.warning("No fields available to validate.")

    with tab4:
        st.subheader("Ask a question about this document")
        question = st.text_input("Your question")
        if question:
            with st.spinner("Generating answer..."):
                answer = answer_question(question, corrected_text)
            st.success(answer)
            st.session_state.history.append((question, answer))

        if st.session_state.history:
            st.subheader("Chat History")
            for q, a in st.session_state.history:
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")

else:
    st.info("Upload a document image or PDF to begin.")
