import re
import unicodedata
from typing import Dict, Any, List
from src.data.claim_parser import PatentClaimParser

class PatentPreprocessor:
    """
    Patent-aware text cleaner and preprocessor.
    Preserves technical terms, numbers, units, symbols, and creates dual representations
    for Transformer models and Lexical indexing.
    """
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Cleans raw text while preserving domain-specific technical entities."""
        if not text or not isinstance(text, str):
            return ""
            
        # 1. Normalize Unicode (NFC)
        text = unicodedata.normalize("NFC", text)
        
        # 2. Remove HTML / XML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # 3. Handle malformed unicode escape characters
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&quot;', '"', text)
        
        # 4. Normalize whitespace (preserve spaces, convert multiple tabs/newlines to single space)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    @classmethod
    def process_patent_record(cls, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a raw patent dictionary record and populates cleaned fields and combined representations.
        """
        title = cls.clean_text(record.get("title", ""))
        abstract = cls.clean_text(record.get("abstract", ""))
        raw_claims = record.get("claims", "")
        
        if isinstance(raw_claims, list):
            claims_str = " ".join([c if isinstance(c, str) else c.get("text", "") for c in raw_claims])
        else:
            claims_str = str(raw_claims or "")
            
        cleaned_claims = cls.clean_text(claims_str)
        independent_claims = PatentClaimParser.extract_independent_claims_text(cleaned_claims)
        
        # Build Transformer Representation: TITLE + ABSTRACT + INDEPENDENT CLAIMS
        transformer_text = f"Title: {title} | Abstract: {abstract} | Claims: {independent_claims}".strip()
        
        # Build Lexical Representation: TITLE + ABSTRACT + ALL CLAIMS
        lexical_text = f"{title} {abstract} {cleaned_claims}".strip()
        
        processed = dict(record)
        processed["title"] = title
        processed["abstract"] = abstract
        processed["claims"] = cleaned_claims
        processed["independent_claims"] = independent_claims
        processed["combined_text_transformer"] = transformer_text
        processed["combined_text_lexical"] = lexical_text
        processed["combined_text"] = transformer_text  # Default alias
        
        return processed
