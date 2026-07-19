import os
import sys
import tempfile

# Inject a mock SQLite file-based database URL for pytest environment 
# so that connection pools can share the database schema.
db_file = os.path.join(tempfile.gettempdir(), "predictiq_test.db")
if os.path.exists(db_file):
    try:
        os.remove(db_file)
    except Exception:
        pass

os.environ["DATABASE_URL"] = f"sqlite:///{db_file}"
os.environ["TESTING"] = "1"

# Ensure the database tables are created in SQLite in-memory before running tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.db.database import init_db
init_db()
