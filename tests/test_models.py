import pytest
from src.models.vader_model import get_vader_scores
from src.models.xgboost_model import predict_onchain

def test_vader_processor():
    texts = ["I love this", "This is terrible"]
    results = get_vader_scores(texts)
    
    assert len(results) == 2
    assert isinstance(results[0], dict)
    assert results[0]['compound'] > 0  # Positivity
    assert results[1]['compound'] < 0  # Negativity

    # Test Empty Data Safety
    empty_results = get_vader_scores([])
    assert len(empty_results) == 0

def test_xgboost_real_classifier():
    assert predict_onchain([0.1, 0.9, 0.5, 50.0]) < 0.5
    assert predict_onchain([0.9, 0.1, 0.1, 1000.0]) > 0.5
    assert predict_onchain([]) > 0.0

from src.models.gnn_model import predict_onchain_gnn

def test_gnn_topology_classifier():
    res1 = predict_onchain_gnn([0.1, 100.0, 50.0, 0.9], interactions=1)
    res2 = predict_onchain_gnn([0.9, 0.5, 2.0, 0.1], interactions=3)
    assert isinstance(res1, float)
    assert isinstance(res2, float)
    assert predict_onchain_gnn([]) == 0.0
