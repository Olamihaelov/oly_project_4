import sqlite3
from passlib.context import CryptContext
import hashlib
import logging
import os

DB_NAME = "users.db"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def row_to_dict(row):
    if row is None:
        return None
    return dict(row)


def hash_password(password: str):
    sha_password = hashlib.sha256(password.encode()).hexdigest()
    return pwd_context.hash(sha_password)


def verify_password(plain_password: str, hashed_password: str):
    sha_password = hashlib.sha256(plain_password.encode()).hexdigest()
    return pwd_context.verify(sha_password, hashed_password)


def create_table_users():
    query = """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_name TEXT NOT NULL UNIQUE,
        email TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL,
        predictions_remaining INTEGER DEFAULT 10,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    with get_connection() as conn:
        conn.execute(query)


def drop_table_users():
    with get_connection() as conn:
        conn.execute("DROP TABLE IF EXISTS users")


def recreate_table_users():
    drop_table_users()
    create_table_users()


def get_all_users():
    query = """
    SELECT id, user_name, email, predictions_remaining
    FROM users
    ORDER BY id
    """
    with get_connection() as conn:
        rows = conn.execute(query).fetchall()
    return [row_to_dict(row) for row in rows]


def get_user_by_id(user_id):
    query = """
    SELECT id, user_name, email, predictions_remaining
    FROM users
    WHERE id = ?
    """
    with get_connection() as conn:
        row = conn.execute(query, (user_id,)).fetchone()
    return row_to_dict(row)


def get_user_by_username(user_name):
    query = """
    SELECT *
    FROM users
    WHERE user_name = ?
    """
    with get_connection() as conn:
        row = conn.execute(query, (user_name,)).fetchone()
    return row_to_dict(row)


def insert_user(user_name, email, password):
    query = """
    INSERT INTO users (user_name, email, password)
    VALUES (?, ?, ?)
    """
    hashed_password = hash_password(password)

    try:
        with get_connection() as conn:
            cursor = conn.execute(query, (user_name, email, hashed_password))
            user_id = cursor.lastrowid
        return get_user_by_id(user_id)
    except sqlite3.IntegrityError:
        return None


def update_user(user_id, user_name, email, password):
    query = """
    UPDATE users
    SET user_name = ?, email = ?, password = ?
    WHERE id = ?
    """
    hashed_password = hash_password(password)

    try:
        with get_connection() as conn:
            cursor = conn.execute(
                query,
                (user_name, email, hashed_password, user_id)
            )
            affected_rows = cursor.rowcount

        if affected_rows == 0:
            return None

        return get_user_by_id(user_id)
    except sqlite3.IntegrityError:
        return "duplicate"


def delete_user(user_id):
    existing_user = get_user_by_id(user_id)
    if existing_user is None:
        return None

    model_filename = f"{existing_user['user_name']}.joblib"
    if os.path.exists(model_filename):
        try:
            os.remove(model_filename)
            logging.info(f"   ✓ Model file deleted: {model_filename}")
        except Exception as e:
            logging.warning(f"   ⚠️ Could not delete model file {model_filename}: {str(e)}")

    with get_connection() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))

    return existing_user


def login_user(user_name, password):
    user = get_user_by_username(user_name)

    if user is None:
        return False

    return verify_password(password, user["password"])


def deduct_prediction(username):
    user = get_user_by_username(username)

    if not user or user["predictions_remaining"] <= 0:
        logging.warning(f"🚫 DEDUCT PREDICTION BLOCKED - No credits remaining | Username: {username}")
        return False

    try:
        with get_connection() as conn:
            conn.execute(
                "UPDATE users SET predictions_remaining = predictions_remaining - 1 WHERE user_name = ?",
                (username,)
            )
        return True
    except Exception as e:
        logging.error(f"Error deducting prediction: {e}")
        return False


def add_predictions(username, amount):
    try:
        with get_connection() as conn:
            conn.execute(
                "UPDATE users SET predictions_remaining = predictions_remaining + ? WHERE user_name = ?",
                (amount, username)
            )
            result = conn.execute(
                "SELECT predictions_remaining FROM users WHERE user_name = ?",
                (username,)
            ).fetchone()
        return result[0] if result else None
    except Exception as e:
        logging.error(f"Error adding predictions: {e}")
        return None


def get_predictions_remaining(username):
    user = get_user_by_username(username)
    return user["predictions_remaining"] if user else None
