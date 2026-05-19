import asyncio
from typing import List, Dict
from src.logger import logger

# Global Singleton tracking
_finbert_pipeline = None

# We use the ProsusAI finbert model explicitly tuned for financial semantic contexts
FINBERT_MODEL_NAME = "ProsusAI/finbert"

def _get_pipeline():
    """Lazy loader to prevent massive PyTorch bootup RAM spikes globally until requested."""
    global _finbert_pipeline
    if _finbert_pipeline is None:
        logger.info(f"Lazy loading {FINBERT_MODEL_NAME} globally. Expect minor startup latency.")
        from transformers import pipeline
        # Use CPU limit natively mapping torch safely
        _finbert_pipeline = pipeline("sentiment-analysis", model=FINBERT_MODEL_NAME, device=-1)
    return _finbert_pipeline

def _run_finbert_batch(texts: List[str]) -> List[float]:
    """Execute within Thread limits. Returns negativity bounds [0.0 -> 1.0]."""
    if not texts:
        return []
        
    try:
        classifier = _get_pipeline()
        
        # Limit texts to avoid extremely heavy memory usage if scraping explodes
        safe_texts = [str(t)[:512] for t in texts] # FinBERT maxes at 512 tokens
        
        # Batch sizes limit the memory mapping internally without blowing out Python constraints
        results = classifier(safe_texts, batch_size=8, truncation=True)
        
        negativity_scores = []
        for r in results:
            if r['label'] == 'negative':
                negativity_scores.append(r['score'])
            elif r['label'] == 'positive':
                negativity_scores.append(0.0) # We only strictly care about fraud/anomaly risks
            else:
                negativity_scores.append(0.05) # Neutral slight bound
                
        return negativity_scores
    except Exception as e:
        logger.error(f"FinBERT execution failed: {e}")
        return [0.0] * len(texts)

async def get_finbert_scores_async(texts: List[str]) -> List[float]:
    """Wraps the synchronous PyTorch processing gracefully off the main asyncio thread loop."""
    if not texts:
        return []
    return await asyncio.to_thread(_run_finbert_batch, texts)
