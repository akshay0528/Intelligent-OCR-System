CATEGORY_KEYWORDS = {
    "Aadhaar Card": ["aadhaar", "uidai", "unique identification", "government of india"],
    "PAN Card": ["permanent account number", "income tax department", "pan card"],
    "Passport": ["passport", "republic of india", "nationality", "place of birth"],
    "Driving License": ["driving licence", "driving license", "transport authority", "rto"],
    "Bank Statement": ["statement of account", "ifsc", "account number", "bank balance"],
    "Bank Passbook": ["passbook", "ifsc", "branch", "account holder"],
    "Resume": ["curriculum vitae", "resume", "education", "experience", "skills", "objective"],
    "Invoice": ["invoice number", "invoice", "gstin", "bill to", "subtotal"],
    "Receipt": ["receipt", "amount paid", "thank you for your purchase", "cashier"],
    "Marksheet": ["marksheet", "grade", "semester", "examination", "cgpa"],
    "Electricity Bill": ["electricity", "consumer number", "units consumed", "kwh"]
}

def classify_document(text):
    if not text or not text.strip():
        return "Other", 0.0

    text_lower = text.lower()
    scores = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches > 0:
            scores[category] = matches / len(keywords)

    if not scores:
        return "Other", 0.0

    best_category = max(scores, key=scores.get)
    confidence = round(min(scores[best_category] * 100, 99.0), 1)
    return best_category, confidence
