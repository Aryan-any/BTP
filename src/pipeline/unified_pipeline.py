import asyncio
from typing import Dict, Any
from src.pipeline.offchain_pipeline import run_offchain_pipeline
from src.pipeline.onchain_pipeline import run_onchain_pipeline
from src.risk.fusion import combine_scores
from src.logger import logger
from src.cache import risk_score_cache
from src.db import update_and_get_trend_async
from src.alerting import trigger_alert

async def detect_scam(project: str) -> Dict[str, Any]:
    """Orchestrates concurrent execution of all analysis pipelines, with TTL Caching and Intelligence Routing."""
    cache_key = project.lower()
    if cache_key in risk_score_cache:
        logger.info(f"CACHE HIT: Returning fast cached results for {project}")
        return risk_score_cache[cache_key]

    logger.info(f"Starting scam detection for project: {project}")
    try:
        # Run pipelines concurrently to vastly improve latency
        offchain_result, onchain_result = await asyncio.gather(
            run_offchain_pipeline(project),
            run_onchain_pipeline(project)
        )
        
        # Unpack Data Paths natively
        path_c_nlp_score, offchain_explanations, confidence_score = offchain_result
        path_a_ml_score, path_b_dl_score = onchain_result
        
        explanations = offchain_explanations
        
        if path_a_ml_score > 0.6:
            explanations.append("XGBoost detected mathematical dataset anomalies.")
        if path_b_dl_score > 0.6:
            explanations.append("Graph DL detected highly connected scam topologies (wash trading).")
        
        # ----------------------------------------------------
        # TRIPLE-ENSEMBLE WEIGHTED VOTING LAYER
        # ----------------------------------------------------
        w_ml = 0.35
        w_dl = 0.30
        w_nlp = 0.35
        
        if confidence_score < 0.3:
            # Shift weight towards rigorous numeric on-chain data if social panic is low
            w_ml += 0.1
            w_dl += 0.1
            w_nlp -= 0.2
        elif confidence_score > 0.8:
            # Significant NLP community panic alerts
            w_nlp += 0.15
            w_ml -= 0.10
            w_dl -= 0.05
            
        final = (w_ml * path_a_ml_score) + (w_dl * path_b_dl_score) + (w_nlp * path_c_nlp_score)
        final_risk = min(max(final, 0.0), 1.0)
        
        # ----------------------------------------------------
        # Intelligence Layer Computations
        # ----------------------------------------------------
        
        # 1. Severity Classification
        if final >= 0.75:
            severity = "CRITICAL"
        elif final >= 0.5:
            severity = "HIGH"
        elif final >= 0.25:
            severity = "MODERATE"
        else:
            severity = "LOW"
            
        # 2. Action Recommendation
        if severity == "CRITICAL":
            action = "AVOID / IMMEDIATE INVESTIGATION"
        elif severity == "HIGH":
            action = "INVESTIGATE BEFORE INVESTING"
        elif severity == "MODERATE":
            action = "MONITOR CLOSELY"
        else:
            action = "LOW RISK"
            
        # 3. Temporal Trend Extraction via Persistent DB
        trend_status = await update_and_get_trend_async(cache_key, final, confidence_score)
        
        # Clean Explanations
        final_explanations = list(dict.fromkeys(explanations))[:5]
        
        # 4. Alert Daemon Trigger
        trigger_alert(project, severity, trend_status, final_explanations)
        
        logger.info(f"Completed analysis for {project}. Final Risk: {final:.4f} | Trend: {trend_status} | Severity: {severity}")
        
        result = {
            "project": project,
            "offchain_risk": round(path_c_nlp_score, 4),
            "onchain_risk": round(max(path_a_ml_score, path_b_dl_score), 4),
            "final_risk": round(final, 4),
            "severity": severity,
            "confidence": round(confidence_score, 4),
            "trend": trend_status,
            "action": action,
            "explanation": final_explanations
        }
        
        # Save to memory cache natively
        risk_score_cache[cache_key] = result
        
        return result
    except Exception as e:
        logger.error(f"Detection failed for {project}: {e}")
        return {
            "project": project,
            "error": "Failed to process analysis",
            "details": str(e)
        }
