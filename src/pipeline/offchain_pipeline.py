import asyncio
from typing import Tuple, List
from src.ingestion.reddit_scraper import get_reddit_data
from src.processing.text_preprocessing import clean_text
from src.models.vader_model import get_vader_scores
from src.models.finbert_model import get_finbert_scores_async
from src.risk.sentiment_risk import compute_sentiment_risk

async def run_offchain_pipeline(keyword: str) -> Tuple[float, List[str], float]:
    try:
        data = await get_reddit_data(keyword)
    except Exception as e:
        from src.logger import logger
        logger.warning(f"Offchain scraping unavailable for {keyword}, gracefully degrading: {e}")
        return 0.0, ["Social footprint NLP currently restricted."], 0.0
        
    cleaned = clean_text(data)
    
    if not cleaned:
        return 0.0, [], 0.0

    # ML inferences across async loop contexts
    vader_task = asyncio.to_thread(get_vader_scores, cleaned)
    finbert_task = get_finbert_scores_async(cleaned)
    
    vader_scores, finbert_scores = await asyncio.gather(vader_task, finbert_task)
    
    # Hybrid 60/40 Injection
    for i, s in enumerate(vader_scores):
        if i < len(finbert_scores):
            s['finbert_neg'] = finbert_scores[i]
            
    return compute_sentiment_risk(vader_scores, data)
