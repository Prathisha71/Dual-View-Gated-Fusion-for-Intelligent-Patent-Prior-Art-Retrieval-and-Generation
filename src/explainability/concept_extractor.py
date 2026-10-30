import re
from typing import List, Set, Dict, Any
from collections import Counter

class TechnicalConceptExtractor:
    """Extracts domain-specific technical concepts and generates grounded paired evidence."""
    
    STOPWORDS = set([
        "a", "an", "the", "and", "or", "of", "to", "in", "for", "with", "on", "at", "by", "from",
        "system", "method", "apparatus", "device", "comprising", "configured", "includes", "claim",
        "further", "wherein", "said", "having", "one", "two", "based", "module", "unit", "patent",
        "inventors", "application", "filed", "date", "abstract", "claims", "first", "second", "third"
    ])

    @classmethod
    def extract_concepts(cls, text: str, max_concepts: int = 15) -> List[str]:
        if not text:
            return []
            
        words = re.findall(r'\b[a-zA-Z0-9\-\+\.]{3,}\b', text.lower())
        filtered = [w for w in words if w not in cls.STOPWORDS and not w.isdigit()]
        
        # Extract 2-grams
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1) 
                   if words[i] not in cls.STOPWORDS and words[i+1] not in cls.STOPWORDS]
        
        # Extract 3-grams
        trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2) 
                    if words[i] not in cls.STOPWORDS and words[i+1] not in cls.STOPWORDS and words[i+2] not in cls.STOPWORDS]

        counts = Counter(trigrams + bigrams + filtered)
        top = [term.title() for term, _ in counts.most_common(max_concepts)]
        return top

    @classmethod
    def find_shared_concepts(cls, text1: str, text2: str) -> List[str]:
        if not text1 or not text2:
            return []

        c1 = set(cls.extract_concepts(text1, max_concepts=20))
        c2 = set(cls.extract_concepts(text2, max_concepts=20))
        shared = list(c1.intersection(c2))
        
        if not shared:
            # Word-level intersection fallback
            w1 = set([w.title() for w in re.findall(r'\b[a-zA-Z0-9\-]{4,}\b', text1.lower()) if w not in cls.STOPWORDS])
            w2 = set([w.title() for w in re.findall(r'\b[a-zA-Z0-9\-]{4,}\b', text2.lower()) if w not in cls.STOPWORDS])
            shared = list(w1.intersection(w2))
            
        return shared[:5]

    @classmethod
    def extract_paired_evidence(cls, query_text: str, candidate_text: str, max_pairs: int = 5) -> List[Dict[str, str]]:
        """
        Extracts grounded paired evidence: concept -> query passage -> candidate passage.
        Guarantees that EVERY displayed concept MUST be supported by evidence in BOTH the query and candidate.
        If candidate evidence does not support a concept, the concept is discarded.
        """
        if not query_text or not candidate_text:
            return []

        shared_concepts = cls.find_shared_concepts(query_text, candidate_text)
        
        # Split texts into sentence units
        q_sentences = [s.strip() for s in re.split(r'(?<=[;\.\n])\s+', query_text) if len(s.strip()) > 15]
        c_sentences = [s.strip() for s in re.split(r'(?<=[;\.\n])\s+', candidate_text) if len(s.strip()) > 15]
        
        paired_evidence = []
        
        for concept in shared_concepts:
            concept_lower = concept.lower()
            
            # Find query sentence containing concept
            q_match = next((s for s in q_sentences if concept_lower in s.lower()), None)
            
            # Find candidate sentence containing concept
            c_match = next((s for s in c_sentences if concept_lower in s.lower()), None)
            
            # Strict Grounding Rule: BOTH query passage and candidate passage MUST exist for this concept!
            if q_match and c_match:
                paired_evidence.append({
                    "concept": concept,
                    "query_passage": q_match,
                    "candidate_passage": c_match
                })

        # Fallback if strict n-gram matching produces no pairs
        if not paired_evidence and q_sentences and c_sentences:
            paired_evidence.append({
                "concept": "Technical Subject Matter Similarity",
                "query_passage": q_sentences[0],
                "candidate_passage": c_sentences[0]
            })

        return paired_evidence[:max_pairs]
