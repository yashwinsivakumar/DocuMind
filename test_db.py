#!/usr/bin/env python
from database import init_db, get_db

init_db()
db = get_db()

# Check tables
tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("Tables:", [t['name'] for t in tables])

# Try inserting a test user
try:
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    password_hash = pwd_context.hash("testpass123")
    
    db.execute(
        "INSERT INTO users (email, password_hash) VALUES (?, ?)",
        ("testuser@example.com", password_hash)
    )
    db.commit()
    print("Test insert successful")
    
    # Verify it was inserted
    user = db.execute("SELECT email FROM users WHERE email = ?", ("testuser@example.com",)).fetchone()
    print(f"Retrieved user: {user['email'] if user else 'not found'}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    db.close()
