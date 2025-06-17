import sys
import os

class HomoLumoReader(object):

    def __init__(self):
        pass

    def find_homo_lumo_energies(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        # Ищем начало блока с энергиями
        start_index = None
        for i, line in enumerate(lines):
            if "ORBITAL ENERGIES" in line:
                start_index = i
        
        if start_index is None:
            return None, None
        
        # Пропускаем разделители и заголовок таблицы
        data_start = start_index + 3
        if data_start >= len(lines):
            return None, None
        
        # Собираем все энергии
        orbitals = []
        for line in lines[data_start:]:
            if not line.strip():  # Пропускаем пустые строки
                continue
            if "---" in line:
                break
            if "*Only the first" in line:
                break
                
            parts = line.split()
            if len(parts) >= 4:
                try:
                    occ = float(parts[1])
                    energy_ev = float(parts[3])
                    orbitals.append((occ, energy_ev))
                except (ValueError, IndexError):
                    continue
        
        if not orbitals:
            return None, None
        
        # Находим HOMO и LUMO
        homo_energy = None
        lumo_energy = None

        for occ, energy in reversed(orbitals):
            if occ > 0:
                homo_energy = energy
                break

        for occ, energy in orbitals:
            if occ == 0:
                lumo_energy = energy
                break
        
        return homo_energy, lumo_energy


    def read_file(self, filename):
        homo, lumo = self.find_homo_lumo_energies(filename)

        if homo is not None and lumo is not None:
            print(f"HOMO energy: {homo:.4f} eV")
            print(f"LUMO energy: {lumo:.4f} eV")
            print(f"HOMO-LUMO gap: {lumo - homo:.4f} eV")

            return [homo, lumo, lumo - homo]
        else:
            print("Orbital energies has not been found in the file")


if __name__ == "__main__":
    print("__name__ == __main__")
else:
    print("homo lumo module has been imported")