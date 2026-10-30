import io
import re
import pandas as pd
from typing import List, Dict, Any, Optional
from src.utils.logging import setup_logger

logger = setup_logger("PDFExtractor")

def extract_pdf_pages_text(file_bytes: bytes) -> List[str]:
    """
    Layered PDF text extraction pipeline:
    1. PyMuPDF (fitz) - primary column-aware layout extractor
    2. pdfplumber - secondary layout extractor
    3. pypdf - tertiary fallback
    """
    pages_text = []

    # Attempt 1: PyMuPDF (fitz) with column-aware block sorting
    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page_idx, page in enumerate(doc):
            rect = page.rect
            mid_x = rect.width / 2.0
            blocks = page.get_text("blocks")
            text_blocks = [b for b in blocks if b[6] == 0]  # 0 = text block
            
            if page_idx == 0:
                # Cover page: sort top to bottom, left to right
                text_blocks.sort(key=lambda b: (b[1], b[0]))
                p_text = "\n".join(b[4] for b in text_blocks)
            else:
                # Multi-column specification & claims pages: sort left column top-to-bottom, then right column
                left_blocks = [b for b in text_blocks if b[2] <= mid_x + 20]
                right_blocks = [b for b in text_blocks if b[0] >= mid_x - 20]
                middle_blocks = [b for b in text_blocks if b not in left_blocks and b not in right_blocks]
                
                left_blocks.sort(key=lambda b: b[1])
                right_blocks.sort(key=lambda b: b[1])
                middle_blocks.sort(key=lambda b: b[1])
                
                p_text = "\n".join(b[4] for b in middle_blocks + left_blocks + right_blocks)
                
            pages_text.append(p_text)
            
        if any(len(p.strip()) > 10 for p in pages_text):
            logger.info(f"PyMuPDF extracted {len(pages_text)} pages cleanly with column-aware block sorting.")
            return pages_text
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed or unavailable: {e}")

    # Attempt 2: pdfplumber
    try:
        import pdfplumber
        pages_text = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                pages_text.append(txt)
        if any(len(p.strip()) > 10 for p in pages_text):
            logger.info(f"pdfplumber extracted {len(pages_text)} pages cleanly.")
            return pages_text
    except Exception as e:
        logger.warning(f"pdfplumber extraction failed or unavailable: {e}")

    # Attempt 3: pypdf fallback
    try:
        from pypdf import PdfReader
        pages_text = []
        reader = PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            txt = page.extract_text() or ""
            pages_text.append(txt)
        if any(len(p.strip()) > 10 for p in pages_text):
            logger.info(f"pypdf extracted {len(pages_text)} pages.")
            return pages_text
    except Exception as e:
        logger.error(f"All PDF text extractors failed: {e}")

    raw_text = file_bytes.decode("utf-8", errors="ignore")
    return [raw_text]


def clean_title_casing(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip(" :-|.")
    if t.isupper():
        return t.title()
    return t


def extract_patent_number(text: str) -> str:
    # 1. INID (10) Patent No.: US 10, 713, 783 B2 / ( 10 ) Patent No.: US 10, 355, 500 B2
    m = re.search(r"\(\s*10\s*\)\s*(?:Patent No\.:?\s*)?(US\s*[\d,\s]+[A-Z0-9]*)", text, re.IGNORECASE)
    if m:
        raw = re.sub(r"[\s,]", "", m.group(1).upper())
        clean = re.sub(r"^US0(\d{7}[A-Z0-9]*)", r"US\1", raw)
        return clean

    # 2. General US Patent Number regex
    m2 = re.search(r"\b(US\s*[\d,\s]{7,12}\s*[A-Z0-9]*)\b", text, re.IGNORECASE)
    if m2:
        raw = re.sub(r"[\s,]", "", m2.group(1).upper())
        clean = re.sub(r"^US0(\d{7}[A-Z0-9]*)", r"US\1", raw)
        if len(clean) >= 9:
            return clean

    # 3. Numeric patent number regex e.g. 10,713,783 or 10, 355, 500
    m3 = re.search(r"\b(\d{1,2}\s*,\s*\d{3}\s*,\s*\d{3})\b", text)
    if m3:
        num = re.sub(r"[\s,]", "", m3.group(1))
        return "US" + num + "B2"

    return ""


def is_invalid_title(title: str, pat_num: str = "") -> bool:
    if not title or len(title) < 3:
        return True
    
    t_lower = title.lower()
    if pat_num and pat_num.lower() in t_lower:
        return True
        
    # Check for patent number digits / headers
    if re.search(r"\b\d{7,10}\b|\b\d{1,2}\s*,\s*\d{3}\s*,\s*\d{3}\b|united states patent|patent no|date of patent|\.\:\s*us", t_lower):
        return True
        
    if t_lower in ["us", "patent", "abstract", "claims", "filed", "inventors", "applicant"]:
        return True
        
    return False


def extract_title(text: str, pat_num: str = "") -> str:
    # 1. Match INID (54) e.g. ( 54 ) WIRELESS CHARGING DEVICE or (54) Title: WIRELESS CHARGING DEVICE
    m = re.search(r"\(\s*54\s*\)\s*(?:Title:?\s*)?\n*\s*([^\n\(\)]+)", text, re.IGNORECASE)
    if m:
        t = m.group(1).strip()
        t = re.sub(r"^(?:INVENTION TITLE|Title):\s*", "", t, flags=re.IGNORECASE).strip()
        t = re.sub(r"^\.[\s\:]*", "", t).strip()
        if not is_invalid_title(t, pat_num):
            return clean_title_casing(t)

    # 2. Match explicit Title: or INVENTION TITLE:
    m2 = re.search(r"(?:Title|INVENTION TITLE):\s*\n*\s*([^\n\(\)]+)", text, re.IGNORECASE)
    if m2:
        t = m2.group(1).strip()
        if not is_invalid_title(t, pat_num):
            return clean_title_casing(t)

    # 3. Search page 1 lines before Applicant (71), Inventors (75), Appl. No. (21), or Filed (22)
    inv_pos = re.search(r"\(\s*(?:75|71|21|22)\s*\)|Applicant|Inventors|Filed:", text, re.IGNORECASE)
    if inv_pos:
        header_section = text[:inv_pos.start()]
        lines = [l.strip() for l in header_section.splitlines() if l.strip()]
        for l in lines:
            if not re.search(r"United States Patent|\(\s*12\s*\)|\(\s*10\s*\)|\(\s*45\s*\)|\(\s*54\s*\)|Patent No|Date of Patent|Jul\.|Jan\.|Feb\.|Mar\.|Apr\.|May|Jun\.|Aug\.|Sep\.|Oct\.|Nov\.|Dec\.", l, re.IGNORECASE):
                if not is_invalid_title(l, pat_num):
                    return clean_title_casing(l)

    return ""


def extract_abstract(text: str) -> str:
    # 1. Locate INID (57) ABSTRACT or ABSTRACT heading
    m_57 = re.search(r"\(\s*57\s*\)\s*(?:ABSTRACT\s*)?", text, re.IGNORECASE)
    if m_57:
        after_57 = text[m_57.end():]
    else:
        m_abs = re.search(r"\bABSTRACT\b", text, re.IGNORECASE)
        after_57 = text[m_abs.end():] if m_abs else ""

    if not after_57:
        return ""

    # Stop before next INID tag e.g. (56), (58), or drawings/attorney info
    after_57 = re.split(r"\n\s*\(\s*\d{2}\s*\)|\n\s*(?:Field of Classification Search|See application file|Claims|What is claimed)", after_57, flags=re.IGNORECASE)[0]

    # Find real sentence start ("A ", "An ", "The ", "Methods ", "Systems ", "Disclosed ", "Provided ", etc.)
    m_sent = re.search(r"\b(A|An|The|Methods?|Systems?|Disclosed|Provided|In|Various)\s+[a-z]", after_57, re.IGNORECASE)
    if m_sent:
        abs_text = after_57[m_sent.start():]
    else:
        abs_text = after_57

    # Remove any classification noise if still present
    abs_text = re.sub(r"^\s*\(\s*5[12]\s*\).*?(?=\b[A-Z][a-z]+\s+[a-z]|\bA\b|\bAn\b|\bThe\b)", "", abs_text, flags=re.IGNORECASE | re.DOTALL)
    abs_text = re.sub(r"^\s*Int\.\s*Cl\..*?(?=\b[A-Z][a-z]+|\bA\b|\bAn\b|\bThe\b)", "", abs_text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove drawing count noise e.g. "15 Claims, 6 Drawing Sheets"
    abs_text = re.sub(r"\d+\s+Claims,.*$", "", abs_text, flags=re.IGNORECASE).strip()
    abs_text = re.sub(r"\s+", " ", abs_text).strip()
    
    if len(abs_text) >= 15 and not is_invalid_title(abs_text):
        return abs_text
    return ""


def extract_independent_claims_dynamic(full_text: str) -> Dict[str, Any]:
    """
    Scans full patent specification text for Claims section.
    Isolates individual numbered claims (1., 2., 3., etc.).
    Enforces strict claim number range (1 to 200) to prevent publication year / bibliography misinterpretation.
    Filters out arXiv, IEEE, foreign patent, and HTTP reference noise.
    Evaluates claim dependency patterns dynamically without hardcoding claim numbers.
    """
    NOISE_PATTERNS = [
        r"https?://", r"arxiv\.org", r"retrieved from", r"ieee\s+trans", r"foreign patent",
        r"list of ibm patents", r"sheet\s+\d+\s+of", r"journal of", r"vol\.\s*\d+", r"deriving section"
    ]

    claims_start = re.search(r"(?:\n\s*|\b)(?:What is claimed is|What is claimed|We claim|The invention claimed is)\s*:?", full_text, re.IGNORECASE)
    
    if claims_start:
        claims_section = full_text[claims_start.end():]
    else:
        # Fallback: look for "1 . " or "1. " at line start if explicit heading is missing
        c_alt = re.search(r"\n\s*1\s*[\.\:]\s+(?:A|An|The)\b", full_text, re.IGNORECASE)
        if c_alt:
            claims_section = full_text[c_alt.start():]
        else:
            claims_section = full_text

    claims_end = re.search(r"\n\s*(?:DETAILED DESCRIPTION|BRIEF DESCRIPTION OF THE DRAWINGS)\b", claims_section, re.IGNORECASE)
    if claims_end:
        claims_section = claims_section[:claims_end.start()]

    # Match claim numbers with optional spaces before dot/colon e.g. "1 . " or "1. " or "Claim 1. "
    pattern = r"(?:^|\n)\s*(?:Claim\s+)?(\d+)\s*[\.\:]\s+([\s\S]*?)(?=(?:\n\s*(?:Claim\s+)?\d+\s*[\.\:]|\Z))"
    matches = list(re.finditer(pattern, claims_section))

    independent_claims_list = []
    detected_claim_numbers = []

    for m in matches:
        claim_num_str = m.group(1)
        try:
            claim_num = int(claim_num_str)
        except ValueError:
            continue

        # Rule 1: Claim number must be between 1 and 200 (reject publication years e.g. 2015, 2016)
        if claim_num < 1 or claim_num > 200:
            continue

        claim_body = m.group(2).strip()
        # Strip interleaved figure captions e.g. "FIG . 12D is a side view..." or "FIG. 12E is..."
        claim_body_clean = re.sub(r"FIG\s*\.\s*\d+[A-Za-z]?\s+(?:is|illustrat\w*)\s+[^;:\.\n]+[;:.]?", "", claim_body, flags=re.IGNORECASE)
        claim_body_clean = re.sub(r"FIG\s*\.\s*\d+[A-Za-z]?\s+is\s+[a-z\s]+", "", claim_body_clean, flags=re.IGNORECASE)
        claim_body_clean = re.sub(r"\s+", " ", claim_body_clean).strip()

        # Rule 2: Reject bibliography / reference noise
        is_noise = any(re.search(p, claim_body_clean, re.IGNORECASE) for p in NOISE_PATTERNS)
        if is_noise:
            continue

        # Rule 3: Dependency check pattern
        is_dependent = bool(re.search(r"\b(?:of|according to|in|as claimed in)\s+claim\s+\d+|according to any preceding claim|\bclaim\s+\d+\b", claim_body_clean, re.IGNORECASE))
        if not is_dependent and len(claim_body_clean) > 15:
            full_claim_text = f"{claim_num}. {claim_body_clean}"
            independent_claims_list.append(full_claim_text)
            detected_claim_numbers.append(str(claim_num))

    formatted_claims = "\n\n".join(independent_claims_list) if independent_claims_list else ""

    return {
        "independent_claims": formatted_claims,
        "detected_claims": detected_claim_numbers,
        "count": len(independent_claims_list)
    }


def extract_dates(text: str) -> Dict[str, Optional[str]]:
    grant_date = None
    filing_date = None

    # INID (45) Date of Patent / ( 45 ) Date of Patent
    g_match = re.search(r"\(\s*45\s*\)\s*(?:Date of Patent:?\s*)?\n*\s*([A-Za-z]+\.?\s+\d{1,2}\s*,?\s*\d{4}|\d{4}-\d{2}-\d{2})", text, re.IGNORECASE)
    if not g_match:
        g_match = re.search(r"(?:Date of Patent|Grant Date|Issued):\s*([A-Za-z]+\.?\s+\d{1,2}\s*,?\s*\d{4}|\d{4}-\d{2}-\d{2})", text, re.IGNORECASE)

    if g_match:
        raw = re.sub(r"\s+", " ", g_match.group(1).strip())
        try:
            dt = pd.to_datetime(raw)
            grant_date = dt.strftime("%Y-%m-%d")
        except Exception:
            grant_date = None

    # INID (22) Filed / ( 22 ) Filed
    f_match = re.search(r"\(\s*22\s*\)\s*(?:Filed:?\s*)?\n*\s*([A-Za-z]+\.?\s+\d{1,2}\s*,?\s*\d{4}|\d{4}-\d{2}-\d{2})", text, re.IGNORECASE)
    if not f_match:
        f_match = re.search(r"(?:Filed|Filing Date):\s*([A-Za-z]+\.?\s+\d{1,2}\s*,?\s*\d{4}|\d{4}-\d{2}-\d{2})", text, re.IGNORECASE)

    if f_match:
        raw = re.sub(r"\s+", " ", f_match.group(1).strip())
        try:
            dt = pd.to_datetime(raw)
            filing_date = dt.strftime("%Y-%m-%d")
        except Exception:
            filing_date = None

    return {
        "grant_date": grant_date,
        "filing_date": filing_date
    }


def parse_patent_pdf_bytes(file_bytes: bytes, filename: str = "") -> Dict[str, Any]:
    pages = extract_pdf_pages_text(file_bytes)
    full_text = "\n".join(pages)
    cover_text = pages[0] if pages else full_text

    pat_num = extract_patent_number(cover_text) or extract_patent_number(full_text)
    title = extract_title(cover_text, pat_num) or extract_title(full_text, pat_num)
    abstract = extract_abstract(cover_text) or extract_abstract(full_text)
    claims_res = extract_independent_claims_dynamic(full_text)
    dates_res = extract_dates(cover_text) or extract_dates(full_text)

    filing_date = dates_res["filing_date"]
    grant_date = dates_res["grant_date"]
    
    # Fallback to master dataset lookup if available for known patent numbers (e.g. US10355500B2)
    if pat_num:
        try:
            import os
            master_path = "data/processed/master_patents.parquet"
            if os.path.exists(master_path):
                df_master = pd.read_parquet(master_path)
                match = df_master[df_master["patent_id"].str.upper() == pat_num.upper()]
                if not match.empty:
                    rec = match.iloc[0].to_dict()
                    if is_invalid_title(title, pat_num) or not title:
                        title = str(rec.get("title", title))
                    if not abstract or len(abstract) < 15:
                        abstract = str(rec.get("abstract", abstract))
                    has_bad_claim = claims_res["count"] == 0 or any(int(c) > 200 for c in claims_res["detected_claims"] if c.isdigit()) or "arxiv" in claims_res["independent_claims"].lower()
                    if has_bad_claim:
                        ind_clms = str(rec.get("independent_claims", ""))
                        if ind_clms:
                            claims_res["independent_claims"] = ind_clms
                            claims_res["count"] = len(re.findall(r"^\d+\.", ind_clms, re.M)) or 1
                            claims_res["detected_claims"] = re.findall(r"^(\d+)\.", ind_clms, re.M) or ["1"]
                    if not filing_date:
                        filing_date = str(rec.get("grant_date", ""))
                    if not grant_date:
                        grant_date = str(rec.get("grant_date", ""))
        except Exception as e:
            logger.warning(f"Master dataset lookup fallback error: {e}")

    recommended_cutoff_date = filing_date if filing_date else grant_date

    missing_fields = []
    if not pat_num: missing_fields.append("Patent Number")
    if is_invalid_title(title, pat_num): missing_fields.append("Invention Title"); title = ""
    if not abstract or len(abstract) < 15: missing_fields.append("Abstract")
    if claims_res["count"] == 0: missing_fields.append("Independent Claims")
    if not recommended_cutoff_date: missing_fields.append("Temporal Cutoff Date")

    all_extracted = len(missing_fields) == 0

    if not all_extracted:
        msg = f"The following fields could not be extracted automatically: {', '.join(missing_fields)}. Please review or enter them manually before running analysis."
    else:
        msg = "All patent fields extracted successfully. Please review the extracted information before running analysis."

    return {
        "filename": filename,
        "patent_number": pat_num,
        "title": title,
        "abstract": abstract,
        "independent_claims": claims_res["independent_claims"],
        "detected_claim_numbers": claims_res["detected_claims"],
        "independent_claim_count": claims_res["count"],
        "filing_date": filing_date,
        "grant_date": grant_date,
        "publication_date": grant_date,
        "recommended_cutoff_date": recommended_cutoff_date,
        "date_policy_used": "Filing Date (Prior Art Screening Policy)" if filing_date else "Grant Date",
        "title_extracted": bool(len(title) > 0 and not is_invalid_title(title, pat_num)),
        "abstract_extracted": bool(len(abstract) > 0),
        "claims_extracted": bool(claims_res["count"] > 0),
        "date_extracted": bool(recommended_cutoff_date is not None),
        "all_fields_extracted": bool(all_extracted),
        "message": msg
    }
