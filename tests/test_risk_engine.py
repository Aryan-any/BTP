import pytest
from src.risk.sentiment_risk import compute_sentiment_risk
from src.db import update_and_get_trend_async
import asyncio

def test_empty_risk_computation():
    """Validates that empty inputs correctly fall back to safe parameters."""
    risk, explanations, confidence = compute_sentiment_risk([])
    
    assert risk == 0.0
    assert len(explanations) == 0
    assert confidence == 0.0

def test_extreme_negative_risk():
    """Validates that a highly negative string generates appropriate risk triggers."""
    scores = [
        {'neg': 0.8, 'neu': 0.1, 'pos': 0.1, 'compound': -0.8},
        {'neg': 0.9, 'neu': 0.1, 'pos': 0.0, 'compound': -0.9}
    ]
    raw_texts = ["dev dumped rug pull", "scam stolen"]
    
    risk, explanations, confidence = compute_sentiment_risk(scores, raw_texts)
    
    assert risk > 0.6
    assert any("Keywords matched" in e for e in explanations)
    assert any("Exceptionally negative" in e for e in explanations)
    assert confidence > 0.0

def test_low_noise_risk():
    """Validates baseline safe computations without triggering panics."""
    scores = [
        {'neg': 0.0, 'neu': 0.9, 'pos': 0.1, 'compound': 0.2},
        {'neg': 0.0, 'neu': 0.8, 'pos': 0.2, 'compound': 0.4}
    ]
    raw_texts = ["looks generally okay", "not sure but fine"]
    
    risk, explanations, confidence = compute_sentiment_risk(scores, raw_texts)
    
    assert risk == 0.0
    assert any("Low volume" in e for e in explanations)

def test_lexicon_without_nlp():
    """Validates Lexicon can drive pure 0-math scores up cleanly over thresholds."""
    scores = [{'neg': 0.0}] * 10 
    raw_texts = ["honeypot rug pull scam"] * 10
    
    risk, explanations, confidence = compute_sentiment_risk(scores, raw_texts)
    
    assert round(risk, 2) == 0.70
    assert any("Keywords matched" in e for e in explanations)

def test_trend_history():
    """Validates the local async db history module logic ensures state limits naturally."""
    import time
    project = f"test_coin_engine_{time.time()}"
    
    async def run_test():
        from src.db import init_db
        await init_db()
        assert await update_and_get_trend_async(project, 0.2, 0.9) == "STABLE"
        assert await update_and_get_trend_async(project, 0.4, 0.9) == "RISING"
        assert await update_and_get_trend_async(project, 0.45, 0.9, threshold=0.1) == "STABLE"
        assert await update_and_get_trend_async(project, 0.1, 0.9) == "FALLING"
        
    asyncio.run(run_test())
