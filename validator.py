import re
from datetime import datetime

def validate_email(value):
    if not value:
        return False
    return bool(re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', str(value)))

def validate_phone(value):
    if not value:
        return False
    digits = re.sub(r'\D', '', str(value))
    return bool(re.match(r'^[6-9]\d{9}$', digits))

def validate_pan(value):
    if not value:
        return False
    return bool(re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', str(value).upper().replace(" ", "")))

def validate_aadhaar(value):
    if not value:
        return False
    digits = re.sub(r'\D', '', str(value))
    return bool(re.match(r'^\d{12}$', digits))

def validate_gst(value):
    if not value:
        return False
    pattern = r'^\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}$'
    return bool(re.match(pattern, str(value).upper().replace(" ", "")))

def validate_ifsc(value):
    if not value:
        return False
    return bool(re.match(r'^[A-Z]{4}0[A-Z0-9]{6}$', str(value).upper()))

def validate_date(value):
    if not value:
        return False
    formats = ["%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d.%m.%Y"]
    for fmt in formats:
        try:
            datetime.strptime(str(value), fmt)
            return True
        except ValueError:
            continue
    return False

def validate_numeric(value):
    if value is None:
        return False
    try:
        float(re.sub(r'[^\d.]', '', str(value)))
        return True
    except ValueError:
        return False

FIELD_VALIDATORS = {
    "email": validate_email,
    "phone": validate_phone,
    "pan": validate_pan,
    "pan_number": validate_pan,
    "aadhaar": validate_aadhaar,
    "aadhaar_number": validate_aadhaar,
    "gst": validate_gst,
    "gst_number": validate_gst,
    "ifsc": validate_ifsc,
    "ifsc_code": validate_ifsc,
    "date": validate_date,
    "date_of_birth": validate_date,
    "dob": validate_date,
    "total_amount": validate_numeric,
    "amount": validate_numeric
}

def validate_fields(extracted_fields):
    results = {}
    if not isinstance(extracted_fields, dict):
        return results

    for field, value in extracted_fields.items():
        key = field.lower().strip()
        validator = FIELD_VALIDATORS.get(key)
        if validator:
            results[field] = {
                "value": value,
                "valid": validator(value)
            }
        else:
            results[field] = {
                "value": value,
                "valid": None
            }

    return results
