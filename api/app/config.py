from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "PRIMED RI Explorer API"
    cors_origins: str = "*"
    nhc_current_storms_url: str = "https://www.nhc.noaa.gov/CurrentStorms.json"
    github_url: str = "https://github.com/m-ahmad-amin/hurricane-ri-forecast"


settings = Settings()
