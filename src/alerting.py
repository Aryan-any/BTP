from typing import List
from src.logger import logger

def trigger_alert(project: str, severity: str, trend: str, explanations: List[str]):
    if severity == "CRITICAL" and trend == "RISING":
        logger.critical(
            f"=== 🚨 ALERT TRIGGERED: {project.upper()} 🚨 ===\n"
            f"Status: {severity} and {trend}\n"
            f"Reasons: {', '.join(explanations)}"
        )
