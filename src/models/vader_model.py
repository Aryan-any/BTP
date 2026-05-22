from typing import List, Dict
import os
import concurrent.futures
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.logger import logger

analyzer = SentimentIntensityAnalyzer()

def _score_single(text: str) -> Dict[str, float]:
    return analyzer.polarity_scores(text)

def get_vader_scores(texts: List[str]) -> List[Dict[str, float]]:
    try:
        if not texts:
            return []
            
        cores = os.cpu_count() or 1
        if len(texts) < 50:
            return [_score_single(t) for t in texts]
            
        logger.info(f"Using {cores} cores to analyze {len(texts)} comments.")
        
        scores = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
            results = executor.map(_score_single, texts, chunksize=max(1, len(texts) // (cores * 4)))
            for res in results:
                scores.append(res)
                
        return scores
    except Exception as e:
        logger.error(f"Error during parallel VADER scoring: {e}")
        return []
