from typing import List, Dict
import os
import concurrent.futures
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.logger import logger

# Initialize globally so subprocesses on Windows import it once
analyzer = SentimentIntensityAnalyzer()

def _score_single(text: str) -> Dict[str, float]:
    """Helper for multiprocessing map."""
    return analyzer.polarity_scores(text)

def get_vader_scores(texts: List[str]) -> List[Dict[str, float]]:
    """Generates sentiment scores using parallel processing for speed over large arrays."""
    try:
        if not texts:
            return []
            
        cores = os.cpu_count() or 1
        # If the texts are sparse, just do it sequentially to save threading overhead
        if len(texts) < 50:
            return [_score_single(t) for t in texts]
            
        logger.info(f"Using {cores} cores to analyze {len(texts)} comments.")
        
        scores = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
            # Map blocks execution until all results are gathered
            results = executor.map(_score_single, texts, chunksize=max(1, len(texts) // (cores * 4)))
            for res in results:
                scores.append(res)
                
        return scores
    except Exception as e:
        logger.error(f"Error during parallel VADER scoring: {e}")
        return []
