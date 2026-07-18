import os
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import declarative_base, sessionmaker

# Database connection setting (Loads environment variable DATABASE_URL if available)
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL:
    # SQLAlchemy requires postgresql:// instead of postgres://
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
else:
    DB_PATH = os.path.join(os.getcwd(), "backend", "database.db")
    DATABASE_URL = f"sqlite:///{DB_PATH}"

# Setup SQLAlchemy engine and SessionLocal
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL, 
        connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative Base
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String, default=lambda: datetime.datetime.now().isoformat())
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

def init_db():
    Base.metadata.create_all(bind=engine)
