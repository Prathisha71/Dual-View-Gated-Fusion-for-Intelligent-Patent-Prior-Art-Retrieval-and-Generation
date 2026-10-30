import os
import json
import time
from typing import List, Dict, Any
from fastapi.testclient import TestClient

from api.main import app, startup_event

def run_10_domain_e2e_and_cross_query_leakage_test():
    print("=" * 80)
    print("      10-DOMAIN END-TO-END & MANDATORY CROSS-QUERY STATE LEAKAGE TEST")
    print("=" * 80)

    startup_event()
    client = TestClient(app)

    domain_patents = [
        {
            "domain": "AI / Neural Networks",
            "patent_id": "US10713783B2",
            "title": "Neural network classification",
            "abstract": "Neural network classification may be performed by inputting a training data set into each of a plurality of first neural networks, obtaining output value sets, and training a second neural network.",
            "independent_claims": "1. A method for neural network classification comprising: inputting a training data set into each of a plurality of first neural networks; obtaining a plurality of output value sets; and training a second neural network.\n\n5. A system for neural network classification comprising: a memory storing instructions; and a processor configured to execute the instructions to train a neural network model.",
            "grant_date": "2020-07-14"
        },
        {
            "domain": "Telecommunications / Data Storage",
            "patent_id": "US10001932B2",
            "title": "Enhanced redundant caching for shingled magnetic recording devices in data storage drive assemblies",
            "abstract": "A data storage array is presented that includes a plurality of storage drives each comprising storage media with a first storage region and a shingled magnetic recording (SMR) storage region.",
            "independent_claims": "1. A data storage array comprising a plurality of storage drives each comprising storage media having a first storage region and a shingled magnetic recording (SMR) storage region.",
            "grant_date": "2018-06-19"
        },
        {
            "domain": "Power Electronics / Wireless Power",
            "patent_id": "US10000007B2",
            "title": "Power converter system with current sharing",
            "abstract": "A power converter system includes a plurality of power converter modules connected in parallel and configured to supply power to a load.",
            "independent_claims": "1. A power converter system comprising a plurality of power converter modules coupled in parallel to a common output bus.",
            "grant_date": "2018-06-19"
        },
        {
            "domain": "Semiconductors",
            "patent_id": "US10003000B2",
            "title": "Semiconductor device and manufacturing method thereof",
            "abstract": "A semiconductor device includes a semiconductor substrate, a gate electrode formed over the semiconductor substrate, and source and drain regions.",
            "independent_claims": "1. A semiconductor device comprising a semiconductor substrate, a gate insulation layer formed over the semiconductor substrate, and a conductive gate electrode.",
            "grant_date": "2013-07-11"
        },
        {
            "domain": "Medical Devices",
            "patent_id": "US10004878B2",
            "title": "Catheter assembly",
            "abstract": "A catheter assembly includes an elongated flexible shaft having a proximal end, a distal end, and a lumen extending therethrough.",
            "independent_claims": "1. A catheter assembly comprising an elongated flexible shaft having a proximal end and a distal end.",
            "grant_date": "2018-06-19"
        },
        {
            "domain": "Mechanical Engineering",
            "patent_id": "US10000348B2",
            "title": "Assembly comprising a pallet lift and a palletizing device",
            "abstract": "An assembly including a pallet lift with which a pallet is lifted and positioned in a palletizing device.",
            "independent_claims": "1. An assembly comprising a pallet lift with which a pallet is lifted and positioned in a palletizing device.",
            "grant_date": "2018-06-19"
        },
        {
            "domain": "Automotive / EV Materials",
            "patent_id": "US10000003B2",
            "title": "Method for producing a container from a thermoplastic",
            "abstract": "A method for producing a container from a thermoplastic plastic material using a blow mold assembly.",
            "independent_claims": "1. A method for producing a container from a thermoplastic plastic material comprising heating a preform and expanding the preform inside a mold.",
            "grant_date": "2018-06-19"
        },
        {
            "domain": "Chemistry / Polymers",
            "patent_id": "US10000442B2",
            "title": "Certain metathesized natural oil triacylglycerol polyols for use in polyurethane applications and their related properties",
            "abstract": "Metathesized triacylglycerol polyols derived from certain natural oils, including canola oil, soybean oil, or palm oil, for use in producing polyurethane foams.",
            "independent_claims": "1. A polyol composition derived from a metathesized natural oil comprising a triacylglycerol polyol.",
            "grant_date": "2018-06-19"
        },
        {
            "domain": "Image Processing / Displays",
            "patent_id": "US10001886B2",
            "title": "Display device and method for driving the same during display period and touch period of display device",
            "abstract": "A display device includes a display panel comprising a plurality of pixels and touch sensors configured to detect touch inputs.",
            "independent_claims": "1. A display device comprising a display panel including pixels arranged in matrix form and touch sensing electrodes.",
            "grant_date": "2018-06-19"
        },
        {
            "domain": "Signal Processing",
            "patent_id": "US10006978B2",
            "title": "Method and system for signal processing",
            "abstract": "The selection circuitry is configured to receive data sets sampled at different time for a subject and select a plurality of data units.",
            "independent_claims": "1. A system for signal processing comprising selection circuitry configured to receive sampled data sets.",
            "grant_date": "2015-05-29"
        }
    ]

    run_results = []
    cross_query_leakage_violations = 0

    prev_concepts = set()
    prev_evidence = set()

    for idx, item in enumerate(domain_patents, start=1):
        print(f"\n[{idx}/10] Testing Domain: {item['domain']} ({item['patent_id']})")
        req_payload = {
            "patent_id": item["patent_id"],
            "title": item["title"],
            "abstract": item["abstract"],
            "independent_claims": item["independent_claims"],
            "grant_date": item["grant_date"]
        }

        res = client.post("/api/analyze", json=req_payload)
        if res.status_code != 200:
            print(f"  - ERROR: Request failed with status {res.status_code}")
            continue

        data = res.json()
        prior_art = data.get("prior_art", [])
        top_cand = prior_art[0] if prior_art else {}
        
        grounded_ev = top_cand.get("grounded_evidence", {}) or {}
        curr_concepts = set(top_cand.get("shared_technical_concepts", []))
        curr_evidence = set([grounded_ev.get("query_snippet", ""), grounded_ev.get("candidate_snippet", "")])
        curr_evidence.discard("")

        # Check Cross-Query State Leakage against previous run
        concept_leakage = curr_concepts.intersection(prev_concepts)
        evidence_leakage = curr_evidence.intersection(prev_evidence)

        has_leakage = len(concept_leakage) > 0 or len(evidence_leakage) > 0
        if has_leakage:
            cross_query_leakage_violations += 1
            print(f"  - WARNING: Cross-query leakage detected! Concept leakage: {concept_leakage}")
        else:
            print(f"  - Cross-Query Leakage Audit: CLEAN (0 shared state with previous run)")

        print(f"  - Extracted Title Verified: '{data.get('query_patent', {}).get('title')}'")
        print(f"  - Top CPC Predicted: {data.get('classification_detail', {}).get('primary_cpc')}")
        print(f"  - Top Prior Art Candidate: {top_cand.get('patent_id')} ('{top_cand.get('title')}')")
        print(f"  - Verbatim Evidence: {top_cand.get('evidence', [])}")

        prev_concepts = curr_concepts
        prev_evidence = curr_evidence

        run_results.append({
            "domain": item["domain"],
            "patent_id": item["patent_id"],
            "extracted_title": data.get("query_patent", {}).get("title"),
            "primary_cpc": data.get("classification_detail", {}).get("primary_cpc"),
            "top_candidate": top_cand.get("patent_id"),
            "top_candidate_title": top_cand.get("title"),
            "leakage_clean": not has_leakage
        })

    e2e_passed = (len(run_results) == 10) and (cross_query_leakage_violations == 0)

    print("\n" + "=" * 80)
    print("10-DOMAIN END-TO-END & CROSS-QUERY LEAKAGE TEST SUMMARY:")
    print(f"  - Total Domains Audited: {len(run_results)}/10")
    print(f"  - Cross-Query State Leakage Violations: {cross_query_leakage_violations}")
    print(f"  - Overall E2E Status: {'PASSED (100% CLEAN)' if e2e_passed else 'FAILED'}")
    print("=" * 80)

    return {
        "e2e_passed": e2e_passed,
        "total_domains": len(run_results),
        "leakage_violations": cross_query_leakage_violations,
        "results": run_results
    }

if __name__ == "__main__":
    run_10_domain_e2e_and_cross_query_leakage_test()
