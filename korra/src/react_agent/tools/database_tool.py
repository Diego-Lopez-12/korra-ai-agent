"""
File: database_tool.py
Author: Your Name
Course: COMPE 475 - Microprocessors
Module: 10 - Human in the Loop
Section: 3
Version: 1.0

Description:
    SQLite database tool for Korra Section 3.
    This file provides CRUD operations for storing files in a database:
        - Save/Create files
        - Read file contents
        - Update existing files
        - Delete files
        - List all files
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import sqlite3
from pathlib import Path
from datetime import datetime
from langchain_core.tools import tool


# ============================================================
# CONFIGURATION CONSTANTS
# ============================================================

DATABASE_NAME = os.getenv(
    "KORRA_DB_PATH",
    str(Path(__file__).resolve().parents[3] / "korra_files.db")
)


# ============================================================
# DATABASE HELPER FUNCTIONS
# ============================================================

def get_connection() -> sqlite3.Connection:
    """
    Create and return a SQLite database connection.

    Returns:
        sqlite3.Connection: Active database connection object
    """
    return sqlite3.connect(DATABASE_NAME)


def initialize_database() -> None:
    """
    Create the files table if it does not already exist.

    Returns:
        None
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT UNIQUE NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()


# ============================================================
# DATABASE TOOL FUNCTIONS
# ============================================================

@tool
def save_file_to_database(filename: str, content: str) -> str:
    """
    Save a new file record to the SQLite database.

    Args:
        filename: Name of the file to save
        content: Text content of the file

    Returns:
        str: Success or error message
    """
    initialize_database()
    conn = get_connection()
    cursor = conn.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        cursor.execute("""
        INSERT INTO files (filename, content, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        """, (filename, content, timestamp, timestamp))

        conn.commit()
        result = f"[SUCCESS] File '{filename}' saved to database."

    except sqlite3.IntegrityError:
        result = f"[ERROR] A file named '{filename}' already exists."

    except sqlite3.Error as e:
        result = f"[ERROR] Database error while saving file: {e}"

    finally:
        conn.close()

    return result


@tool
def read_file_from_database(filename: str) -> str:
    """
    Retrieve a file's contents from the SQLite database.

    Args:
        filename: Name of the file to read

    Returns:
        str: File contents or error message
    """
    initialize_database()
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
        SELECT filename, content, created_at, updated_at
        FROM files
        WHERE filename = ?
        """, (filename,))

        result = cursor.fetchone()

        if result is None:
            return f"[ERROR] File '{filename}' was not found in the database."

        stored_filename, content, created_at, updated_at = result

        return (
            f"[SUCCESS] File found.\n"
            f"Filename: {stored_filename}\n"
            f"Created At: {created_at}\n"
            f"Updated At: {updated_at}\n"
            f"Content:\n{content}"
        )

    except sqlite3.Error as e:
        return f"[ERROR] Database error while reading file: {e}"

    finally:
        conn.close()


@tool
def update_file_in_database(filename: str, new_content: str) -> str:
    """
    Update an existing file's content in the SQLite database.

    Args:
        filename: Name of the file to update
        new_content: New text content to store

    Returns:
        str: Success or error message
    """
    initialize_database()
    conn = get_connection()
    cursor = conn.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        cursor.execute("""
        SELECT id FROM files
        WHERE filename = ?
        """, (filename,))
        existing_file = cursor.fetchone()

        if existing_file is None:
            return f"[ERROR] File '{filename}' does not exist."

        cursor.execute("""
        UPDATE files
        SET content = ?, updated_at = ?
        WHERE filename = ?
        """, (new_content, timestamp, filename))

        conn.commit()
        return f"[SUCCESS] File '{filename}' updated successfully."

    except sqlite3.Error as e:
        return f"[ERROR] Database error while updating file: {e}"

    finally:
        conn.close()


@tool
def delete_file_from_database(filename: str) -> str:
    """
    Delete a file record from the SQLite database.

    Args:
        filename: Name of the file to delete

    Returns:
        str: Success or error message
    """
    initialize_database()
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
        SELECT id FROM files
        WHERE filename = ?
        """, (filename,))
        existing_file = cursor.fetchone()

        if existing_file is None:
            return f"[ERROR] File '{filename}' does not exist."

        cursor.execute("""
        DELETE FROM files
        WHERE filename = ?
        """, (filename,))

        conn.commit()
        return f"[SUCCESS] File '{filename}' deleted successfully."

    except sqlite3.Error as e:
        return f"[ERROR] Database error while deleting file: {e}"

    finally:
        conn.close()


@tool
def list_files_in_database() -> str:
    """
    List all files currently stored in the SQLite database.

    Returns:
        str: Formatted list of files or error message
    """
    initialize_database()
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
        SELECT filename, created_at, updated_at
        FROM files
        ORDER BY filename
        """)

        results = cursor.fetchall()

        if not results:
            return "[SUCCESS] The database is empty. No files are currently stored."

        output_lines = ["[SUCCESS] Here is the complete list of files currently stored in the database:"]
        for row in results:
            filename, created_at, updated_at = row
            output_lines.append(
                f"- {filename} | Created: {created_at} | Updated: {updated_at}"
            )

        return "\n".join(output_lines)

    except sqlite3.Error as e:
        return f"[ERROR] Database error while listing files: {e}"

    finally:
        conn.close()