import aiosqlite
from typing import List, Tuple
from src.logger import logger

DB_PATH = "scam_data.db"

async def init_db():
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute('PRAGMA journal_mode=WAL;')
            await db.execute('PRAGMA synchronous=NORMAL;')
            await db.execute('''
                CREATE TABLE IF NOT EXISTS trends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    risk REAL,
                    confidence REAL
                )
            ''')
            await db.commit()
    except Exception as e:
        logger.error(f"Database Initialization Failed: {e}")

async def write_history(project: str, risk: float, confidence: float):
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT INTO trends (project, risk, confidence) VALUES (?, ?, ?)",
                (project, risk, confidence)
            )
            await db.commit()
    except Exception as e:
        logger.error(f"Failed to persist string: {e}")

async def fetch_recent_risks(project: str, limit: int = 10) -> List[float]:
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute(
                "SELECT risk FROM trends WHERE project = ? ORDER BY id DESC LIMIT ?",
                (project, limit)
            ) as cursor:
                rows = await cursor.fetchall()
                # Return in chronological order (earliest to latest from the slice)
                risks = [row[0] for row in rows]
                return risks[::-1]
    except Exception as e:
        logger.error(f"Failed to read from DB: {e}")
        return []

async def update_and_get_trend_async(project: str, current_risk: float, confidence: float, threshold: float = 0.1) -> str:
    # New payload
    await write_history(project, current_risk, confidence)
    
    history = await fetch_recent_risks(project, limit=2) # Only need the last two specifically for simple threshold delta
    
    if len(history) < 2:
        return "STABLE"
        
    previous_risk = history[0]
    delta = current_risk - previous_risk
    
    if delta > threshold:
        return "RISING"
    elif delta < -threshold:
        return "FALLING"
    else:
        return "STABLE"
