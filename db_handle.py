import sqlite3

class DataBase(object):

    def __init__(self):
        self.conn = sqlite3.connect('database.db')
        self.cursor = self.conn.cursor()
        self.create_homo_lumo_data_table()

    def __del__(self):
        self.conn.close()

    def create_homo_lumo_data_table(self):
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS homo_lumo_data (
            long_name TEXT NOT NULL,
            short_name TEXT NOT NULL,
            functional TEXT NOT NULL,
            homo_energy REAL,
            lumo_energy REAL,
            homo_lumo_gap REAL,
            PRIMARY KEY (short_name, functional)
        )
        ''')

        self.conn.commit()


    def add_homo_lumo_data(self, long_name: str, short_name: str, functional: str, homo_energy: float, lumo_energy: float):
        self.cursor.execute("INSERT OR REPLACE INTO homo_lumo_data (long_name, short_name, functional, homo_energy, lumo_energy, homo_lumo_gap) VALUES (?, ?, ?, ?, ?, ?)",
                            (long_name, short_name, functional, homo_energy, lumo_energy, lumo_energy-homo_energy))
        self.conn.commit()


    def get_homo_lumo_data(self, functional: str, parameter: str):
        self.cursor.execute(f"SELECT short_name, {parameter} FROM homo_lumo_data WHERE functional='{functional}'")
        return self.cursor.fetchall()