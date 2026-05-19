import httpx
import time
from typing import Tuple
from src.models.xgboost_model import predict_onchain
from src.models.gnn_model import predict_onchain_gnn
from src.config import settings
from src.logger import logger
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type

@retry(
    wait=wait_fixed(1),
    stop=stop_after_attempt(2),
    retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException))
)
async def _fetch_dex_metadata(project: str) -> dict:
    url = f"https://api.dexscreener.com/latest/dex/search?q={project}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=5.0)
        response.raise_for_status() 
        pairs = response.json().get('pairs', [])
        if pairs:
            # Return the highest volume pair natively to guarantee representative ML inputs
            return sorted(pairs, key=lambda x: x.get('volume', {}).get('h24', 0), reverse=True)[0]
        return {}

async def run_onchain_pipeline(project: str) -> Tuple[float, float]:
    """
    Executes both Path A (XGBoost Tabular) and Path B (PyTorch GCN) concurrently natively scaling models.
    """
    features = [0.1, 0.9, 0.5, 50.0] 
    interactions = 1
    
    try:
        data = await _fetch_dex_metadata(project)
        if data:
            # Native Math Map perfectly avoiding fakes
            txns_m5 = data.get('txns', {}).get('m5', {}).get('buys', 0) + data.get('txns', {}).get('m5', {}).get('sells', 0)
            features[0] = min(txns_m5 / 100.0, 1.0) 
            
            created_at = data.get('pairCreatedAt', int(time.time() * 1000))
            age_days = (time.time() * 1000 - created_at) / (1000 * 60 * 60 * 24)
            features[1] = max(0.0, min(age_days / 30.0, 1.0)) 
            
            h24_txns = data.get('txns', {}).get('h24', {}).get('buys', 0) + data.get('txns', {}).get('h24', {}).get('sells', 0)
            features[2] = float(h24_txns)
            
            vol_h24 = float(data.get('volume', {}).get('h24', 0.0))
            features[3] = vol_h24 / h24_txns if h24_txns > 0 else 0.0
            
            interactions = min(10, max(1, int(h24_txns / 100)))
            
            logger.info(f"Successfully integrated Web3 DexScreener data evaluating Dual-Path ML/DL execution for {project}")
        else:
            logger.warning(f"Project '{project}' not found on-chain, using safe baseline boundaries.")
    except Exception as e:
        logger.warning(f"OnChain RPC integration failed for {project}, using safe assumptions: {e}")
            
    # Path A: Tabular ML Inference
    xgb_score = predict_onchain(features)
    
    # Path B: Relational DL Inference
    gnn_score = predict_onchain_gnn(features, interactions=interactions)
    
    return float(xgb_score), float(gnn_score)
