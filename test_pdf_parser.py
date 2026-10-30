import re
import pandas as pd

def parse_uspto_patent_text(text: str) -> dict:
    patent_number = ""
    title = ""
    abstract = ""
    claims = ""
    grant_date = ""
    filing_date = ""

    # 1. Patent Number: INID (10) or US Patent No.
    pat_match = re.search(r"\((?:10)\)\s*(?:Patent No\.:?\s*)?(US\s*[\d,]+\s*[A-Z0-9]*)", text, re.IGNORECASE)
    if pat_match:
        patent_number = re.sub(r"[\s,]", "", pat_match.group(1).upper())
    else:
        pat_match2 = re.search(r"US\s*([\d,]{7,10})\s*([A-Z0-9]*)", text, re.IGNORECASE)
        if pat_match2:
            patent_number = "US" + pat_match2.group(1).replace(",", "") + pat_match2.group(2)
        else:
            patent_number = ""

    # 2. Title: INID (54)
    title_match = re.search(r"\((?:54)\)\s*(?:Title:?\s*)?([^\n\(\)]+)", text, re.IGNORECASE)
    if title_match:
        title = title_match.group(1).strip()
        title = re.sub(r"^(?:INVENTION TITLE|Title):\s*", "", title, flags=re.IGNORECASE).strip()
    else:
        title_match2 = re.search(r"(?:Title|INVENTION TITLE):\s*([^\n]+)", text, re.IGNORECASE)
        if title_match2:
            title = title_match2.group(1).strip()
        else:
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            valid = [l for l in lines[:15] if not re.search(r"United States Patent|\(12\)|\(10\)|Patent No|Date of Patent|\(57\)", l, re.IGNORECASE)]
            title = valid[0] if valid else "Extracted Invention Title"

    if title.isupper():
        title = title.capitalize()

    # 3. Grant Date: INID (45) Date of Patent
    grant_match = re.search(r"\((?:45)\)\s*(?:Date of Patent:?\s*)?([A-Za-z]+\.?\s+\d{1,2},\s*\d{4}|\d{4}-\d{2}-\d{2})", text, re.IGNORECASE)
    if grant_match:
        try:
            dt = pd.to_datetime(grant_match.group(1).strip())
            grant_date = dt.strftime("%Y-%m-%d")
        except Exception:
            grant_date = ""
    else:
        grant_match2 = re.search(r"(?:Date of Patent|Grant Date):\s*([A-Za-z]+\.?\s+\d{1,2},\s*\d{4}|\d{4}-\d{2}-\d{2})", text, re.IGNORECASE)
        if grant_match2:
            try:
                dt = pd.to_datetime(grant_match2.group(1).strip())
                grant_date = dt.strftime("%Y-%m-%d")
            except Exception:
                grant_date = ""

    # 4. Filing Date: INID (22) Filed
    filing_match = re.search(r"\((?:22)\)\s*(?:Filed:?\s*)?([A-Za-z]+\.?\s+\d{1,2},\s*\d{4}|\d{4}-\d{2}-\d{2})", text, re.IGNORECASE)
    if filing_match:
        try:
            dt = pd.to_datetime(filing_match.group(1).strip())
            filing_date = dt.strftime("%Y-%m-%d")
        except Exception:
            filing_date = ""
    else:
        filing_match2 = re.search(r"(?:Filed|Filing Date):\s*([A-Za-z]+\.?\s+\d{1,2},\s*\d{4}|\d{4}-\d{2}-\d{2})", text, re.IGNORECASE)
        if filing_match2:
            try:
                dt = pd.to_datetime(filing_match2.group(1).strip())
                filing_date = dt.strftime("%Y-%m-%d")
            except Exception:
                filing_date = ""

    # 5. Abstract: INID (57) ABSTRACT
    abs_match = re.search(r"\((?:57)\)\s*(?:ABSTRACT\s*)?(.*?)(?=\n\s*\(\d{2}\)|\n\s*[A-Z\s]{4,}:|What is claimed is|Claims|$)", text, re.IGNORECASE | re.DOTALL)
    if abs_match:
        abstract = abs_match.group(1).strip()
        abstract = re.sub(r"^\s*ABSTRACT\s*", "", abstract, flags=re.IGNORECASE).strip()
    else:
        abs_match2 = re.search(r"ABSTRACT\s*\n+(.*?)(?=\n\n|\n\s*[A-Z\s]{4,}:|What is claimed is|Claims|$)", text, re.IGNORECASE | re.DOTALL)
        if abs_match2:
            abstract = abs_match2.group(1).strip()
        else:
            abstract = ""

    # 6. Claims & Independent Claims Extraction
    claims_section = ""
    claims_start = re.search(r"(?:What is claimed is|What is claimed|CLAIMS)\s*:?", text, re.IGNORECASE)
    if claims_start:
        claims_section = text[claims_start.end():]
    else:
        claims_section = text

    claim_blocks = re.findall(r"(?:^|\n)\s*(\d+)\.\s+([\s\S]*?)(?=(?:\n\s*\d+\.|\Z))", claims_section)

    independent_claims_list = []
    for claim_num, claim_text in claim_blocks:
        clean_text = claim_text.strip()
        is_dependent = bool(re.search(r"\b(?:of|according to|in|as claimed in)\s+claim\s+\d+", clean_text, re.IGNORECASE))
        if not is_dependent:
            independent_claims_list.append(f"{claim_num}. {clean_text}")

    if independent_claims_list:
        claims = "\n\n".join(independent_claims_list)
    else:
        claims = ""

    return {
        "patent_number": patent_number,
        "title": title,
        "abstract": abstract,
        "independent_claims": claims,
        "grant_date": grant_date,
        "filing_date": filing_date,
        "independent_claim_count": len(independent_claims_list)
    }

# Test sample USPTO text
sample_uspto_text = """
(12) United States Patent
(10) Patent No.: US 10,713,783 B2
(45) Date of Patent: Jul. 14, 2020
(54) NEURAL NETWORK CLASSIFICATION
(71) Applicant: Google LLC
(22) Filed: Jun. 1, 2017
(57) ABSTRACT
Neural network classification may be performed by inputting a training data set into each of a plurality of first neural networks, obtaining output values, and training a second neural network.

What is claimed is:
1. A method for neural network classification comprising: inputting a training data set into each of a plurality of first neural networks; obtaining a plurality of output value sets; and training a second neural network.
2. The method of claim 1, further comprising initializing network weights.
3. The method of claim 1, wherein the training data set includes image samples.
4. The method of claim 1, further comprising normalizing input features.
5. A system for neural network classification comprising: a memory storing instructions; and a processor configured to execute the instructions to train a neural network model.
6. The system of claim 5, further comprising a graphics processing unit.
7. A non-transitory computer-readable storage medium storing instructions for neural network classification that, when executed by a processor, cause the processor to perform operations comprising training a second neural network.
"""

if __name__ == "__main__":
    res = parse_uspto_patent_text(sample_uspto_text)
    print("Parsed Results:")
    print("Patent Number:", res["patent_number"])
    print("Title:", res["title"])
    print("Grant Date:", res["grant_date"])
    print("Filing Date:", res["filing_date"])
    print("Abstract:", res["abstract"])
    print("Independent Claim Count:", res["independent_claim_count"])
    print("Independent Claims:\n", res["independent_claims"])
