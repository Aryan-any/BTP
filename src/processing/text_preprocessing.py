import re
from typing import List
from src.logger import logger

# Compile regex patterns once for efficiency
HTTP_PATTERN = re.compile(r"http\S+")
NON_ALPHANUM_PATTERN = re.compile(r"[^A-Za-z0-9\s!?.]")

def clean_text(texts: List[str]) -> List[str]:
    """Cleans a list of strings by removing URLs and non-alphanumeric characters."""
    try:
        if not texts:
            return []
            
        return [
            NON_ALPHANUM_PATTERN.sub("", HTTP_PATTERN.sub("", text)).lower()
            for text in texts
        ]
    except Exception as e:
        logger.error(f"Error during text preprocessing: {e}")
        return []
