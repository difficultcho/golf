from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application configuration settings"""

    # Server
    APP_NAME: str = "Video Processing API"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = True

    # File paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_VIDEOS_DIR: Path = DATA_DIR / "raw_videos"
    PROCESSED_VIDEOS_DIR: Path = DATA_DIR / "processed_videos"
    METADATA_DIR: Path = DATA_DIR / "metadata"

    # Phase 2A directories
    POSE_DATA_DIR: Path = DATA_DIR / "pose_data"
    ANALYSIS_RESULTS_DIR: Path = DATA_DIR / "analysis_results"
    VISUALIZATION_DIR: Path = DATA_DIR / "visualizations"

    # Video constraints
    MAX_VIDEO_SIZE_MB: int = 100
    ALLOWED_VIDEO_FORMATS: set = {".mp4", ".mov", ".avi", ".webm"}

    # Processing
    DUMMY_PROCESSING_DELAY: float = 1.0  # Simulate processing time (seconds)

    # Phase 2A: Analysis settings
    MJCF_MODEL_PATH: str = "assets/mjcf/humanoid_golf.xml"
    POSE_CONFIDENCE_THRESHOLD: float = 0.5  # MediaPipe detection threshold
    ANALYSIS_FPS: int = 30  # Target FPS for analysis

    # Database (MySQL)
    MYSQL_HOST: str = "127.0.0.1"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = ""
    MYSQL_DATABASE: str = "golf"

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"mysql+aiomysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
        )

    # WeChat Mini Program
    WECHAT_APPID: str = ""
    WECHAT_SECRET: str = ""

    # JWT
    JWT_SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_HOURS: int = 168  # 7 days

    # CORS
    ALLOWED_ORIGINS: list = ["*"]  # Restrict in production

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
