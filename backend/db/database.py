import os
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import declarative_base, sessionmaker

# Load environment variables manually from .env if present
for env_name in [".env", "backend/.env"]:
    dotenv_path = os.path.join(os.getcwd(), env_name)
    if os.path.exists(dotenv_path):
        try:
            with open(dotenv_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        vk = v.strip()
                        # Strip single or double quotes around value if present
                        if (vk.startswith('"') and vk.endswith('"')) or (vk.startswith("'") and vk.endswith("'")):
                            vk = vk[1:-1]
                        os.environ[k.strip()] = vk
        except Exception:
            pass

# Database connection setting (Loads environment variable DATABASE_URL if available)
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable is required to connect to Supabase/PostgreSQL. "
        "SQLite support has been disabled. Please set DATABASE_URL in your environment or .env file."
    )

# SQLAlchemy requires postgresql:// instead of postgres://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Setup SQLAlchemy engine and SessionLocal for Supabase/PostgreSQL
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative Base
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String, default=lambda: datetime.datetime.now().isoformat())
    customer_id = Column(String, nullable=True)
    recency = Column(Float)
    frequency = Column(Float)
    monetary = Column(Float)
    avg_order_value = Column(Float)
    unique_products = Column(Float)
    return_rate = Column(Float)
    customer_lifetime_days = Column(Float)
    purchase_frequency_monthly = Column(Float)
    avg_quantity_per_order = Column(Float)
    predicted_segment = Column(String)
    lr_pred = Column(Integer)
    lr_prob = Column(Float)
    rf_pred = Column(Integer)
    rf_prob = Column(Float)
    xgb_pred = Column(Integer)
    xgb_prob = Column(Float)
    lgb_pred = Column(Integer)
    lgb_prob = Column(Float)
    stacking_pred = Column(Integer)
    stacking_prob = Column(Float)

class DriftMetric(Base):
    __tablename__ = "drift_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String, default=lambda: datetime.datetime.now().isoformat())
    feature_name = Column(String)
    psi_value = Column(Float)
    status = Column(String)
    training_mean = Column(Float)
    production_mean = Column(Float)

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=True)
    hashed_password = Column(String, nullable=False)

class ModelMetadata(Base):
    __tablename__ = "model_metadata"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String, default=lambda: datetime.datetime.now().isoformat())
    model_name = Column(String)
    filepath = Column(String)
    accuracy = Column(Float)
    roc_auc = Column(Float)
    f1_score = Column(Float)
    mcc = Column(Float)
    is_active = Column(Integer, default=0) # 1 if active, 0 otherwise

def init_db():
    Base.metadata.create_all(bind=engine)

