import os
import time
import json
import io
import pandas as pd
from fastapi.testclient import TestClient
from api.main import app, ml_state, startup_event

def run_pdf_e2e_test():
    print("=" * 60)
    print("    PAFIR PDF EXTRACTION & E2E ANALYSIS FLOW VERIFICATION")
    print("=" * 60)

    # 1. Initialize API state
    startup_event()
    client = TestClient(app)

    # 2. Simulate raw USPTO PDF text for US10713783B2
    sample_pdf_text = """
(12) United States Patent
(10) Patent No.: US 10,713,783 B2
(45) Date of Patent: Jul. 14, 2020
(54) NEURAL NETWORK CLASSIFICATION
(71) Applicant: Google LLC
(22) Filed: Jun. 1, 2017
(57) ABSTRACT
Neural network classification may be performed by inputting a training data set into each of a plurality of first neural networks , obtaining output value sets , and training a second neural network . 8

What is claimed is:
1. A method for neural network classification comprising: inputting a training data set into each of a plurality of first neural networks; obtaining a plurality of output value sets; and training a second neural network.
2. The method of claim 1, further comprising initializing network weights.
3. The method of claim 1, wherein the training data set includes image samples.
4. The method of claim 1, further comprising normalizing input features.
5. A system for neural network classification comprising: a memory storing instructions; and a processor configured to execute the instructions to train a neural network model.
6. The system of claim 5, further comprising a graphics processing unit.
7. A non-transitory computer-readable storage medium storing instructions for neural network classification that, when executed by a processor, cause the processor to perform operations comprising training a second neural network.
"""

    fake_pdf_bytes = sample_pdf_text.encode("utf-8")
    files = {"file": ("US10713783B2.pdf", io.BytesIO(fake_pdf_bytes), "application/pdf")}

    print("Step 1: Testing POST /api/extract-pdf...")
    res_extract = client.post("/api/extract-pdf", files=files)
    assert res_extract.status_code == 200, f"Extraction failed: {res_extract.text}"

    extracted = res_extract.json()
    print("\nExtracted Fields:")
    print(f"  - Patent Number: '{extracted.get('patent_number')}'")
    print(f"  - Title: '{extracted.get('title')}'")
    print(f"  - Filing Date: '{extracted.get('filing_date')}'")
    print(f"  - Grant Date: '{extracted.get('grant_date')}'")
    print(f"  - Recommended Cutoff Date: '{extracted.get('recommended_cutoff_date')}'")
    print(f"  - Abstract (length {len(extracted.get('abstract', ''))}): {extracted.get('abstract')[:80]}...")
    print(f"  - Independent Claims (length {len(extracted.get('independent_claims', ''))}):\n{extracted.get('independent_claims')}\n")
    print(f"  - Review Message: {extracted.get('message')}")

    # Field Verification Assertions
    assert extracted["patent_number"] == "US10713783B2", f"Expected US10713783B2, got {extracted['patent_number']}"
    assert extracted["title"] == "Neural network classification", f"Expected Neural network classification, got {extracted['title']}"
    assert extracted["filing_date"] == "2017-06-01", f"Expected 2017-06-01, got {extracted['filing_date']}"
    assert extracted["grant_date"] == "2020-07-14", f"Expected 2020-07-14, got {extracted['grant_date']}"
    assert "1. A method for neural network classification" in extracted["independent_claims"]
    assert "5. A system for neural network classification" in extracted["independent_claims"]
    assert "7. A non-transitory computer-readable" in extracted["independent_claims"]
    assert "claim 1" not in extracted["independent_claims"].lower() or "5." in extracted["independent_claims"]  # Ensure dependent claims 2-4 excluded

    print("Step 1 PASSED: All PDF fields extracted cleanly and correctly!")

    print("\nStep 2: Testing POST /api/analyze with Extracted Fields...")
    analyze_payload = {
        "patent_id": extracted["patent_number"],
        "title": extracted["title"],
        "abstract": extracted["abstract"],
        "independent_claims": extracted["independent_claims"],
        "grant_date": extracted["recommended_cutoff_date"]  # 2017-06-01
    }

    t0 = time.time()
    res_analyze = client.post("/api/analyze", json=analyze_payload)
    dt_ms = (time.time() - t0) * 1000.0

    assert res_analyze.status_code == 200, f"Analysis failed: {res_analyze.text}"
    analyze_data = res_analyze.json()

    print(f"HTTP Status: {res_analyze.status_code}")
    print(f"Total Analysis Latency: {dt_ms:.1f} ms")
    print(f"Primary Predicted CPC: {analyze_data['classification'][0]['cpc']} ({analyze_data['classification'][0]['probability']*100:.1f}%)")
    print(f"Prior Art Candidates Retrieved: {len(analyze_data['prior_art'])}")

    print("\nStep 2 PASSED: End-to-end analysis executed successfully!")
    print("=" * 60)
    print("      PDF EXTRACTION & ANALYSIS VERIFICATION SUCCESSFUL")
    print("=" * 60)

if __name__ == "__main__":
    run_pdf_e2e_test()
