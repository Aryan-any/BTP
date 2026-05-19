import asyncpraw
import asyncio
from typing import List
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from src.config import settings
from src.logger import logger

async def _fetch_comments_from_submission(submission) -> List[str]:
    """Helper to heavily parallelize the actual HTTP comment fetching part."""
    comments = []
    # Avoid replacing more comments to stay lightning fast
    await submission.load()
    for comment in submission.comments:
        if isinstance(comment, asyncpraw.models.Comment):
            if hasattr(comment, "body") and comment.body:
                comments.append(comment.body)
    return comments

# Retry logic: Wait 2s, then 4s, maximum 3 retries, only for Exceptions (prevents networking crashes)
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(Exception)
)
async def get_reddit_data(keyword: str, limit: int = 5) -> List[str]:
    """Asynchronously fetches newest Reddit comments for a given keyword using async gathers."""
    reddit = None
    
    try:
        reddit = asyncpraw.Reddit(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            user_agent=settings.reddit_user_agent
        )

        subreddit = await reddit.subreddit("CryptoCurrency")
        
        # Async generator for searching - Sort specifically by 'new' for near-real-time context!
        tasks = []
        async for submission in subreddit.search(keyword, sort='new', limit=limit):
            tasks.append(_fetch_comments_from_submission(submission))
            
        # Fire off all comment loads concurrently
        gathered_comments = await asyncio.gather(*tasks)
        
        # Flatten the list of lists
        comments = [item for sublist in gathered_comments for item in sublist]
                        
        return comments
    except Exception as e:
        logger.error(f"Error fetching data from Reddit for '{keyword}': {e}")
        # Reraise so tenacity catches it
        raise e
    finally:
        if reddit:
            await reddit.close()
