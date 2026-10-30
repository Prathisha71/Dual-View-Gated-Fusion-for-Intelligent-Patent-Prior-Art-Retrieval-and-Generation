import re
from typing import List, Dict, Tuple

class PatentClaimParser:
    """
    Parser for patent claims. Distinguishes independent claims from dependent claims,
    extracts claim numbers, preserves technical terminology, units, numbers, and symbols.
    """
    
    @staticmethod
    def parse_claims(claims_text: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Parses raw patent claim text or formatted claim string into independent and dependent claims.
        """
        if not claims_text or not isinstance(claims_text, str):
            return {"independent": [], "dependent": [], "all": []}
            
        # Split by claim numbers (e.g. "1.", "2.", "Claim 1:", "1. A device comprising...")
        raw_chunks = re.split(r'(?:\r?\n|\s)+(?=(\d+)\.\s+|Claim\s+(\d+)[:\s])', claims_text, flags=re.IGNORECASE)
        
        # Alternative split if not structured with numbered headings
        if len(raw_chunks) <= 1:
            lines = [line.strip() for line in claims_text.split('\n') if line.strip()]
            claim_blocks = []
            curr_block = []
            for line in lines:
                if re.match(r'^\d+\.\s+', line):
                    if curr_block:
                        claim_blocks.append(" ".join(curr_block))
                        curr_block = []
                curr_block.append(line)
            if curr_block:
                claim_blocks.append(" ".join(curr_block))
        else:
            claim_blocks = [chunk.strip() for chunk in raw_chunks if chunk and chunk.strip()]
            
        independent_claims = []
        dependent_claims = []
        all_claims = []
        
        # Regex pattern for dependent claim references
        dependent_pattern = re.compile(
            r'\b(?:according to|claim|claims|of claim|in claim)\s+\d+\b', 
            re.IGNORECASE
        )
        
        for idx, block in enumerate(claim_blocks):
            if not block or len(block) < 5:
                continue
                
            # Extract claim number
            num_match = re.match(r'^(?:Claim\s+)?(\d+)[\.\:]?\s*(.*)', block, re.DOTALL | re.IGNORECASE)
            if num_match:
                claim_num = num_match.group(1)
                text = num_match.group(2).strip()
            else:
                claim_num = str(idx + 1)
                text = block.strip()
                
            is_dependent = bool(dependent_pattern.search(text))
            claim_obj = {
                "claim_number": claim_num,
                "text": text,
                "is_independent": not is_dependent
            }
            
            all_claims.append(claim_obj)
            if is_dependent:
                dependent_claims.append(claim_obj)
            else:
                independent_claims.append(claim_obj)
                
        return {
            "independent": independent_claims,
            "dependent": dependent_claims,
            "all": all_claims
        }

    @staticmethod
    def extract_independent_claims_text(claims_text: str) -> str:
        """Utility to get concatenated text of all independent claims."""
        parsed = PatentClaimParser.parse_claims(claims_text)
        if parsed["independent"]:
            return " ".join([c["text"] for c in parsed["independent"]])
        elif parsed["all"]:
            # Fallback to first claim if dependency check fails
            return parsed["all"][0]["text"]
        return claims_text or ""
