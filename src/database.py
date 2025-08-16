import sqlite3

def create_connection(db_file="images.db"):
    """ Create a database connection to the SQLite database. """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)
    return conn