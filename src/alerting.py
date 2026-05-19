from typing import List
from src.logger import logger

def trigger_alert(project: str, severity: str, trend: str, explanations: List[str]):
    """
    Checks pipeline thresholds for critical events that should be pushed to end-users 
    or external hooks natively avoiding silent failures.
    """
    if severity == "CRITICAL" and trend == "RISING":
        logger.critical(
            f"=== 🚨 ALERT TRIGGERED: {project.upper()} 🚨 ===\n"
            f"Status: {severity} and {trend}\n"
            f"Reasons: {', '.join(explanations)}"
        )
        # Optional: Add webhook POST trigger logic natively here mapping payload to Discord/Telegram
