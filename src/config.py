from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    reddit_client_id: str
    reddit_client_secret: str
    reddit_user_agent: str
    moralis_api_key: str = "" # Optional Web3
    port: int = 8000
    environment: str = "development"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
