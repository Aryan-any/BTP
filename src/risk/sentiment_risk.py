import numpy as np
import re
import math
from typing import List, Dict, Tuple
from src.logger import logger

CRYPTO_FRAUD_LEXICON = [
    r"\brug\s?pull\b", r"\bhoneypot\b", r"\bscam\b", r"\bponzi\b",
    r"\bdevs?\s?dump(ed)?\b", r"\bcontract\s?drain\b", r"\bslow\s?rug\b",
    r"\bwarning\b", r"\bsteal\b", r"\bstolen\b", r"\bexploit\b",
    r"\bmint\s?function\b", r"\bliquidity\s?unlocked\b", r"\bdev\s?wallet\s?dumped\b",
    r"\bblacklisted\s?address\b", r"\bcan\'?t\s?sell\b", r"\bsiphoned\b"
]

FRAUD_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in CRYPTO_FRAUD_LEXICON]

def _calculate_lexicon_risk(raw_texts: List[str]) -> Tuple[float, List[str]]:
    if not raw_texts:
        return 0.0, []
        
    hits = 0
    matched_flags = set()
    for text in raw_texts:
        for pattern in FRAUD_PATTERNS:
            match = pattern.search(text)
            if match:
                hits += 1
                matched_flags.add(match.group(0).lower())
                break # Only count 1 major hit per comment to avoid saturation
                
    hit_ratio = hits / len(raw_texts)
    lexicon_risk = min(hit_ratio * 5.0, 1.0) 
    
    explanations = []
    if matched_flags:
        flags_str = ', '.join(list(matched_flags)[:3])
        if len(matched_flags) > 3:
            flags_str += "..."
        explanations.append(f"Keywords matched: {flags_str}")
        
    return lexicon_risk, explanations

def compute_sentiment_risk(scores: List[Dict[str, float]], raw_texts: List[str] = None) -> Tuple[float, List[str], float]:
    explanations = []
    try:
        volume = len(scores) if scores else 0
        confidence = min(1.0, math.log(volume + 1) / 5.0)

        if volume == 0:
            logger.warning("No sentiment scores provided. Returning default safe boundaries.")
            return 0.0, explanations, confidence

        negatives = []
        for s in scores:
            vader_n = s.get('neg', 0.0)
            finbert_n = s.get('finbert_neg', vader_n) 
            hybrid = (0.6 * finbert_n) + (0.4 * vader_n)
            negatives.append(hybrid)

        avg_neg = float(np.mean(negatives))
        velocity = float(np.std(negatives)) if volume > 1 else 0.0

        nlp_risk = avg_neg * velocity * (volume / 100)
        
        if velocity > 0.15:
            explanations.append("High negative sentiment spike detected")
        if avg_neg > 0.2:
            explanations.append("Exceptionally negative baseline sentiment ratio")
        
        lexicon_risk, lexicon_explanations = _calculate_lexicon_risk(raw_texts) if raw_texts else (0.0, [])
        explanations.extend(lexicon_explanations)
        
        composite_risk = (nlp_risk * 0.3) + (lexicon_risk * 0.7)
        final_risk = min(max(composite_risk, 0.0), 1.0)
        
        if volume < 10:
            explanations.append("Low volume anomaly detected (sparse data)")
        elif volume > 150:
            explanations.append("Unusual discussion volume observed")
            
        unique_expl = list(dict.fromkeys(explanations))
        final_explanations = unique_expl[:5]
            
        return final_risk, final_explanations, confidence
    except Exception as e:
        logger.error(f"Error computing sentiment risk: {e}")
        return 0.0, ["Failed to calculate internal risk (error recovery)"], 0.0
