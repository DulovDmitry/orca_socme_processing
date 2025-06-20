import sqlite3

class DataBase(object):

    def __init__(self):
        self.conn = sqlite3.connect('database.db')
        self.cursor = self.conn.cursor()
        self.create_homo_lumo_data_table()
        self.create_singlet_triplet_data_table()
        self.create_socme_data_table()
        self.create_multiwfn_data_table()

    def __del__(self):
        self.conn.close()


    '''
    The functions that create tables
    '''
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

    def create_singlet_triplet_data_table(self):
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS singlet_triplet_data (
            long_name TEXT NOT NULL,
            short_name TEXT NOT NULL,
            functional TEXT NOT NULL,
            S1_energy REAL,
            T1_energy REAL,
            delta_E_S1_T1 REAL,
            delta_E_S2_T1 REAL,
            PRIMARY KEY (short_name, functional)
        )
        ''')

        self.conn.commit()

    def create_socme_data_table(self):
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS socme_data (
            long_name TEXT NOT NULL,
            short_name TEXT NOT NULL,
            functional TEXT NOT NULL,
            T1_S1_SOCME REAL,
            T1_S2_SOCME REAL,
            T1_S1_kRISC REAL,
            T1_S2_kRISC REAL,
            PRIMARY KEY (short_name, functional)
        )
        ''')

        self.conn.commit()

    def create_multiwfn_data_table(self):
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS multiwfn_data (
            short_name TEXT PRIMARY KEY NOT NULL,
            distance REAL,
            integral_norm REAL,
            integral_square REAL
        )
        ''')

        self.conn.commit()


    '''
    The functions that add data to the tables
    '''
    def add_homo_lumo_data(self, long_name: str, short_name: str, functional: str, homo_energy: float, lumo_energy: float):
        self.cursor.execute("INSERT OR REPLACE INTO homo_lumo_data (long_name, short_name, functional, homo_energy, lumo_energy, homo_lumo_gap) VALUES (?, ?, ?, ?, ?, ?)",
                            (long_name, short_name, functional, homo_energy, lumo_energy, lumo_energy-homo_energy))
        self.conn.commit()

    def add_singlet_triplet_data(self, long_name: str, short_name: str, functional: str, S1_energy: float, T1_energy: float, delta_E_S1_T1: float, delta_E_S2_T1: float):
        self.cursor.execute("INSERT OR REPLACE INTO singlet_triplet_data (long_name, short_name, functional, S1_energy, T1_energy, delta_E_S1_T1, delta_E_S2_T1) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (long_name, short_name, functional, S1_energy, T1_energy, delta_E_S1_T1, delta_E_S2_T1))
        self.conn.commit()

    def add_socme_data(self, long_name: str, short_name: str, functional: str, T1_S1_SOCME: float, T1_S2_SOCME: float, T1_S1_kRISC: float, T1_S2_kRISC: float):
        self.cursor.execute("INSERT OR REPLACE INTO socme_data (long_name, short_name, functional, T1_S1_SOCME, T1_S2_SOCME, T1_S1_kRISC, T1_S2_kRISC) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (long_name, short_name, functional, T1_S1_SOCME, T1_S2_SOCME, T1_S1_kRISC, T1_S2_kRISC))
        self.conn.commit()

    def add_multiwfn_data(self, short_name: str, distance: float, integral_norm: float, integral_square: float):
        self.cursor.execute("INSERT OR REPLACE INTO multiwfn_data (short_name, distance, integral_norm, integral_square) VALUES (?, ?, ?, ?)",
                            (short_name, distance, integral_norm, integral_square))
        self.conn.commit()

    '''
    The functions that get data from the tables
    '''
    def get_homo_lumo_data(self, functional: str, parameter: str):
        self.cursor.execute(f"SELECT short_name, {parameter} FROM homo_lumo_data WHERE functional='{functional}'")
        return self.cursor.fetchall()

    def get_singlet_triplet_data(self, functional: str, parameter: str):
        self.cursor.execute(f"SELECT short_name, {parameter} FROM singlet_triplet_data WHERE functional='{functional}'")
        return self.cursor.fetchall()

    def get_socme_data(self, functional: str, parameter: str):
        self.cursor.execute(f"SELECT short_name, {parameter} FROM socme_data WHERE functional='{functional}'")
        return self.cursor.fetchall()

    def get_multiwfn_data(self, parameter: str):
        self.cursor.execute(f"SELECT short_name, {parameter} FROM multiwfn_data")
        return self.cursor.fetchall()