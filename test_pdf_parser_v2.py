import re
import pandas as pd

def clean_title_casing(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip(" :-|")
    if t.isupper():
        return t.capitalize()
    return t

def extract_patent_number(text: str) -> str:
    m = re.search(r"\((?:10)\)\s*(?:Patent No\.:?\s*)?(US\s*[\d,]+\s*[A-Z0-9]*)", text, re.IGNORECASE)
    if m:
        raw = re.sub(r"[\s,]", "", m.group(1).upper())
        clean = re.sub(r"^US0(\d{7}[A-Z0-9]*)", r"US\1", raw)
        return clean

    m2 = re.search(r"US\s*([\d,]{7,10})\s*([A-Z0-9]*)", text, re.IGNORECASE)
    if m2:
        raw = "US" + m2.group(1).replace(",", "") + m2.group(2).upper()
        clean = re.sub(r"^US0(\d{7}[A-Z0-9]*)", r"US\1", raw)
        return clean

    return ""

def extract_title(text: str) -> str:
    m = re.search(r"\((?:54)\)\s*(?:Title:?\s*)?\n*\s*([^\n\(\)]+)", text, re.IGNORECASE)
    if m:
        t = m.group(1).strip()
        t = re.sub(r"^(?:INVENTION TITLE|Title):\s*", "", t, flags=re.IGNORECASE).strip()
        if not re.search(r"^US\s*\d+|United States Patent|Patent No", t, re.IGNORECASE) and len(t) > 3:
            return clean_title_casing(t)

    m2 = re.search(r"(?:Title|INVENTION TITLE):\s*\n*\s*([^\n\(\)]+)", text, re.IGNORECASE)
    if m2:
        t = m2.group(1).strip()
        if not re.search(r"^US\s*\d+|United States Patent|Patent No", t, re.IGNORECASE) and len(t) > 3:
            return clean_title_casing(t)

    inv_pos = re.search(r"\((?:75|71|21|22)\)", text)
    if inv_pos:
        header_section = text[:inv_pos.start()]
        lines = [l.strip() for l in header_section.splitlines() if l.strip()]
        for l in lines:
            if not re.search(r"United States Patent|\(12\)|\(10\)|\(45\)|Patent No|Date of Patent|Jul\.|Jan\.|Feb\.|Mar\.|Apr\.|May|Jun\.|Aug\.|Sep\.|Oct\.|Nov\.|Dec\.", l, re.IGNORECASE):
                if not re.match(r"^\(?US\s*[\d,]+[\w]*\)?$", l, re.IGNORECASE) and len(l) > 3:
                    return clean_title_casing(l)

    return ""

def extract_abstract(text: str) -> str:
    abs_match = re.search(r"\((?:57)\)\s*(?:ABSTRACT\s*)?(.*?)(?=\n\s*\(\d{2}\)|\n\s*[A-Z\s]{4,}:|What is claimed is|Claims|$)", text, re.IGNORECASE | re.DOTALL)
    if not abs_match:
        abs_match = re.search(r"ABSTRACT\s*\n+(.*?)(?=\n\n|\n\s*[A-Z\s]{4,}:|What is claimed is|Claims|$)", text, re.IGNORECASE | re.DOTALL)

    if abs_match:
        raw_abs = abs_match.group(1).strip()
        clean = re.sub(r"^\s*ABSTRACT\s*", "", raw_abs, flags=re.IGNORECASE).strip()
        clean = re.sub(r"(?:\s+\.\s+|\s+)\d+\s*$", "", clean)
        clean = re.sub(r"(\w+)\s+ting\b", r"\1ting", clean)
        clean = re.sub(r"\s+([\.,;])", r"\1", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean
    return ""

def extract_independent_claims(text: str) -> str:
    claims_start = re.search(r"(?:What is claimed is|What is claimed|CLAIMS)\s*:?", text, re.IGNORECASE)
    if not claims_start:
        return ""

    claims_section = text[claims_start.end():]
    claims_end = re.search(r"\n\s*(?:DETAILED DESCRIPTION|BRIEF DESCRIPTION OF THE DRAWINGS)\b", claims_section, re.IGNORECASE)
    if claims_end:
        claims_section = claims_section[:claims_end.start()]

    pattern = r"(?:^|\n)\s*(\d+)\.\s+([\s\S]*?)(?=(?:\n\s*\d+\.|\Z))"
    matches = list(re.finditer(pattern, claims_section))

    independent_claims = []
    for m in matches:
        claim_num = m.group(1)
        claim_body = m.group(2).strip()
        claim_body_clean = re.sub(r"\s+", " ", claim_body)

        is_dependent = bool(re.search(r"\b(?:of|according to|in|as claimed in)\s+claim\s+\d+", claim_body_clean, re.IGNORECASE))
        if not is_dependent:
            full_claim_text = f"{claim_num}. {claim_body_clean}"
            independent_claims.append(full_claim_text)

    if independent_claims:
        return "\n\n".join(independent_claims)
    return ""

def extract_dates(text: str) -> dict:
    grant_date = ""
    filing_date = ""

    g_match = re.search(r"\((?:45)\)\s*(?:Date of Patent:?\s*)?\n*\s*([A-Za-z]+\.?\s+\d{1,2},\s*\d{4}|\d{4}-\d{2}-\d{2})", text, re.IGNORECASE)
    if not g_match:
        g_match = re.search(r"(?:Date of Patent|Grant Date):\s*([A-Za-z]+\.?\s+\d{1,2},\s*\d{4}|\d{4}-\d{2}-\d{2})", text, re.IGNORECASE)

    if g_match:
        raw = g_match.group(1).strip()
        try:
            dt = pd.to_datetime(raw)
            grant_date = dt.strftime("%Y-%m-%d")
        except Exception:
            grant_date = ""

    f_match = re.search(r"\((?:22)\)\s*(?:Filed:?\s*)?\n*\s*([A-Za-z]+\.?\s+\d{1,2},\s*\d{4}|\d{4}-\d{2}-\d{2})", text, re.IGNORECASE)
    if not f_match:
        f_match = re.search(r"(?:Filed|Filing Date):\s*([A-Za-z]+\.?\s+\d{1,2},\s*\d{4}|\d{4}-\d{2}-\d{2})", text, re.IGNORECASE)

    if f_match:
        raw = f_match.group(1).strip()
        try:
            dt = pd.to_datetime(raw)
            filing_date = dt.strftime("%Y-%m-%d")
        except Exception:
            filing_date = ""

    return {
        "grant_date": grant_date,
        "filing_date": filing_date
    }

def parse_pdf_text_full(text: str) -> dict:
    pat_num = extract_patent_number(text)
    title = extract_title(text)
    abstract = extract_abstract(text)
    claims = extract_independent_claims(text)
    dates = extract_dates(text)

    rec_date = dates["filing_date"] if dates["filing_date"] else dates["grant_date"]

    missing_fields = []
    if not pat_num: missing_fields.append("Patent Number")
    if not title: missing_fields.append("Invention Title")
    if not abstract: missing_fields.append("Abstract")
    if not claims: missing_fields.append("Independent Claims")
    if not rec_date: missing_fields.append("Temporal Cutoff Date")

    if missing_fields:
        message = f"The following fields could not be extracted automatically: {', '.join(missing_fields)}. Please review or enter them manually."
    else:
        message = "All patent fields extracted successfully. Please review the extracted information before running analysis."

    return {
        "patent_number": pat_num,
        "title": title,
        "abstract": abstract,
        "independent_claims": claims,
        "grant_date": dates["grant_date"],
        "filing_date": dates["filing_date"],
        "recommended_cutoff_date": rec_date,
        "message": message,
        "all_fields_extracted": len(missing_fields) == 0
    }

# Test sample with newline after (54)
test_text = """
(12) United States Patent
(10) Patent No.: US 10,713,783 B2
(45) Date of Patent: Jul. 14, 2020
(54)
NEURAL NETWORK CLASSIFICATION
(71) Applicant: Google LLC
(22) Filed: Jun. 1, 2017
(57) ABSTRACT
Neural network classification may be performed by inputting a training data set into each of a plurality of first neural networks , obtaining output value sets , and training a second neural network . 8

What is claimed is:
1. A method for neural network classification comprising: inputting a training data set into each of a plurality of first neural networks; obtaining output value sets; and training a second neural network.
2. The method of claim 1, further comprising initializing network weights.
5. A system for neural network classification comprising: a memory storing instructions; and a processor configured to execute the instructions to train a neural network model.
6. The system of claim 5, further comprising a graphics processing unit.
7. A non-transitory computer-readable storage medium storing instructions for neural network classification that, when executed by a processor, cause the processor to perform operations comprising training a second neural network.
"""

if __name__ == "__main__":
    res = parse_pdf_text_full(test_text)
    print("=== TEST PARSING RESULTS ===")
    print("Patent Number:", res["patent_number"])
    print("Title:", res["title"])
    print("Grant Date:", res["grant_date"])
    print("Filing Date:", res["filing_date"])
    print("Recommended Cutoff Date:", res["recommended_cutoff_date"])
    print("Abstract:\n", res["abstract"])
    print("Independent Claims:\n", res["independent_claims"])
    print("Message:\n", res["message"])
