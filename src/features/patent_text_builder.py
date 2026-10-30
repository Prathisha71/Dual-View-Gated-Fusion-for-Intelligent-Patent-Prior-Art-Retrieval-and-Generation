from typing import Dict, Any, Tuple

class PatentTextBuilder:
    """Helper to format patent text representations for Context View and Claim View."""
    
    @staticmethod
    def get_context_text(record: Dict[str, Any]) -> str:
        title = record.get("title", "").strip()
        abstract = record.get("abstract", "").strip()
        return f"Title: {title} | Abstract: {abstract}".strip()

    @staticmethod
    def get_claim_text(record: Dict[str, Any]) -> str:
        independent_claims = record.get("independent_claims", "").strip()
        if not independent_claims:
            claims = record.get("claims", "").strip()
            return f"Independent Claims: {claims}".strip()
        return f"Independent Claims: {independent_claims}".strip()

    @classmethod
    def extract_views(cls, record: Dict[str, Any]) -> Tuple[str, str]:
        return cls.get_context_text(record), cls.get_claim_text(record)
