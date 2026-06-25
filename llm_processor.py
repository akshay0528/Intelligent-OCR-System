import torch
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

_model = None
_tokenizer = None
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

def load_model():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if torch.cuda.is_available():
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
    return _model, _tokenizer

def run_llm(prompt, max_new_tokens=512):
    model, tokenizer = load_model()
    messages = [
        {"role": "system", "content": "You are a precise document analysis assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    return response.strip()

def correct_ocr_text(raw_text):
    if not raw_text.strip():
        return raw_text

    prompt = (
        "Correct OCR errors in the following extracted document text. "
        "Fix spacing, broken words, and obvious character misreads. "
        "Do not add new information. Return only the corrected text.\n\n"
        f"OCR Text:\n{raw_text}"
    )
    try:
        return run_llm(prompt, max_new_tokens=250)
    except Exception:
        return raw_text

def extract_structured_fields(corrected_text, doc_type):
    prompt = (
        f"The following text was extracted from a document classified as '{doc_type}'. "
        "Extract the relevant fields and return ONLY a valid JSON object, no explanation, "
        "no markdown formatting, no code fences. If a field is not present, use null.\n\n"
        "Use field names appropriate to the document type, for example: "
        "name, date_of_birth, document_number, address, phone, email, "
        "invoice_number, total_amount, date, issuer.\n\n"
        f"Document Text:\n{corrected_text}\n\nJSON:"
    )
    try:
        response = run_llm(prompt, max_new_tokens=250)
        return parse_json_response(response)
    except Exception as e:
        return {"error": f"LLM extraction failed: {str(e)}"}

def correct_and_extract(raw_text, doc_type):
    if not raw_text.strip():
        return raw_text, {}

    prompt = (
        "You will be given raw OCR text from a document classified as "
        f"'{doc_type}'. Do two things:\n"
        "1. Mentally correct OCR errors (spacing, broken words, misreads).\n"
        "2. Extract relevant fields from the corrected text as JSON "
        "(name, date_of_birth, document_number, address, phone, email, "
        "invoice_number, total_amount, date, issuer — use null if absent).\n\n"
        "Return your answer in EXACTLY this format, nothing else:\n"
        "CORRECTED:\n<corrected text>\nJSON:\n<json object>\n\n"
        f"OCR Text:\n{raw_text}"
    )
    try:
        response = run_llm(prompt, max_new_tokens=450)
        corrected = raw_text
        fields = {}

        if "CORRECTED:" in response and "JSON:" in response:
            corrected_part = response.split("CORRECTED:")[1].split("JSON:")[0].strip()
            json_part = response.split("JSON:")[1].strip()
            corrected = corrected_part if corrected_part else raw_text
            fields = parse_json_response(json_part)
        else:
            fields = parse_json_response(response)

        return corrected, fields
    except Exception as e:
        return raw_text, {"error": f"LLM processing failed: {str(e)}"}

def parse_json_response(response):
    cleaned = re.sub(r"```json|```", "", response).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group()
    try:
        return json.loads(cleaned)
    except Exception:
        return {"raw_response": cleaned}

def answer_question(question, document_text):
    prompt = (
        "Answer the question using only the information in the document text below. "
        "If the answer is not present, say 'Not found in document.' Be concise.\n\n"
        f"Document Text:\n{document_text}\n\nQuestion: {question}\nAnswer:"
    )
    try:
        return run_llm(prompt, max_new_tokens=150)
    except Exception as e:
        return f"Error generating answer: {str(e)}"
