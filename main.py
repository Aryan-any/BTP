import asyncio
from src.pipeline.unified_pipeline import detect_scam

async def main():
    print("Running test execution for 'FTX'...")
    result = await detect_scam("PEPE")
    print("\nResult:")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
