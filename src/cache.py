from cachetools import TTLCache

# Cache up to 1000 items, with each expiring in 300 seconds (5 minutes)
risk_score_cache = TTLCache(maxsize=1000, ttl=300)
