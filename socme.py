import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

# Глобальные переменные
triplet_levels_output_file = "triplet_levels.txt"
start_marker_triplets = "TD-DFT/TDA EXCITED STATES (TRIPLETS)"
end_marker_triplets = "\n\n---"

singlet_levels_output_file = "singlet_levels.txt"
start_marker_singlets = "TD-DFT/TDA EXCITED STATES (SINGLETS)"
end_marker_singlets = "***"

socme_output_file = "socme.txt"
start_marker_socme = "CALCULATED SOCME BETWEEN TRIPLETS AND SINGLETS"
end_marker_socme = "SOC stabilization of the ground state"

number_of_singlets = 0

### Настройки matplotlib
scaling_factor = 1.2
fig = plt.figure(figsize=(16*scaling_factor, 9*scaling_factor))  # общий размер фигуры

# Создаем сетку 2x1, где первый график будет в 5 раз выше второго
gs = gridspec.GridSpec(2, 2, height_ratios=[5, 1])

# Создаем субплоты
ax1 = fig.add_subplot(gs[0:3])  # график heatmat
ax2 = fig.add_subplot(gs[1])  # энергретическая диаграмма
ax3 = fig.add_subplot(gs[3])  # таблица с основными параметрами


def read_complex_matrix(filename):
    # Инициализация данных
    data = []
    max_T = 0
    max_S = 0
    
    with open(filename, 'r') as file:
        for line in file:
            # Пропускаем служебные строки
            stripped = line.strip()
            if not stripped or '---' in stripped or 'Root' in stripped or 'T      S' in stripped or "CALCULATED" in stripped:
                continue
            
            # Разбиваем строку на компоненты
            parts = re.split(r'[()]', stripped)
            if len(parts) < 7:  # Должно быть 3 комплекса (Z, X, Y)
                continue
            
            # Извлекаем T и S
            ts_part = parts[0].strip()
            ts_match = re.match(r'^\s*(\d+)\s+(\d+)', ts_part)
            if not ts_match:
                continue
                
            T = int(ts_match.group(1))
            S = int(ts_match.group(2))
            max_T = max(max_T, T)
            max_S = max(max_S, S)
            
            # Обрабатываем все три комплексных числа
            complexes = []
            for i in range(1, 6, 2):  # Индексы 1,3,5 для Z,X,Y
                num_part = parts[i].strip()
                num_match = re.match(r'\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)', num_part)
                if num_match:
                    real = float(num_match.group(1))
                    imag = float(num_match.group(2))
                    complexes.append(complex(real, imag))
            
            if len(complexes) == 3:
                # Вычисляем общий модуль
                total_module = np.sqrt(sum(abs(c)**2 for c in complexes))
                data.append((T, S, total_module))
    
    # Создаем матрицу (индексы начинаются с 1)
    matrix = np.zeros((max_T + 1, max_S + 1))
    for T, S, module in data:
        matrix[T, S] = module
    
    return matrix


def plot_heatmap(matrix):
    #plt.figure()
    #plt.subplot(1,2,1)
    
    # Маскируем нулевые значения
    # mask = matrix == 0

    # Не маскируем нулевые значения
    mask = matrix == -1
    
    # Находим минимальное и максимальное ненулевые значения
    non_zero_values = matrix[matrix > 0]
    if len(non_zero_values) == 0:
        print("No non-zero values found in the matrix!")
        return
    
    vmin = np.min(non_zero_values)
    vmax = np.max(matrix)
    
    # Создаем подписи для осей
    s_labels = [f'S{i}' for i in range(matrix.shape[1])]
    t_labels = [f'T{i}' for i in range(1, matrix.shape[0])]  # T начинается с 1
    
    # Инвертируем матрицу по вертикали, чтобы T1 был снизу
    inverted_matrix = np.flipud(matrix[1:,:])
    inverted_mask = np.flipud(mask[1:,:])
    inverted_t_labels = t_labels[::-1]  # Инвертируем порядок меток
    
    # Рисуем heatmap
    heatmap_ax = sns.heatmap(
        inverted_matrix,
        cmap='plasma',
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        mask=inverted_mask,
        vmin=vmin,
        vmax=vmax,
        xticklabels=s_labels,
        yticklabels=inverted_t_labels,
        cbar=False,  # Это убирает цветовую шкалу
        ax=ax1
    )

    # Настройки графика
    ax1.set_title(r'Spin-Orbit Coupling Matrix Elements, $cm^{-1}$', pad=20, fontsize=14)
    ax1.set_xlabel('Singlet States', fontsize=12)
    ax1.set_ylabel('Triplet States', fontsize=12)
    
    # Улучшаем читаемость подписей
    ax1.tick_params(axis='both', rotation=0, which='major', labelsize=10)


def extract_fragment(input_file, output_file, start_marker, end_marker):
    """
    Извлекает фрагмент текста между start_marker и end_marker из input_file и сохраняет в output_file.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            content = file.read()
        
        start_index = content.find(start_marker)
        if start_index == -1:
            print(f"Маркер начала '{start_marker}' не найден.")
            return
        
        # Ищем конец фрагмента (начало следующего блока)
        end_index = content.find(end_marker, start_index)
        if end_index == -1:
            print(f"Маркер конца '{end_marker}' не найден. Сохраняем до конца файла.")
            fragment = content[start_index:]
        else:
            fragment = content[start_index:end_index].strip()
        
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(fragment)
        
        print(f"Фрагмент успешно сохранён в {output_file}")
    
    except FileNotFoundError:
        print(f"Файл {input_file} не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


def process_socme_file():
    """
    Парсит файл со значениями SOCME
    """
    try:
        print(f"Processing file: {socme_output_file}")
        result_matrix = read_complex_matrix(socme_output_file)
        
        print("\nMatrix of complex number modules (T x S):")
        print(np.round(result_matrix, 4))
        
        print("\nPlotting heatmap...")
        plot_heatmap(result_matrix)
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    else:
        return result_matrix[1][1]


def extract_energy_levels(text):
    """
    Извлекает номера состояний и соответствующие энергии в eV из текста
    """
    pattern = r'STATE\s+(\d+):\s+E=\s+(\d+\.\d+)'
    matches = re.findall(pattern, text)
    states = []
    energies_eV = []
    for match in matches:
        states.append(int(match[0]))
        energies_eV.append(float(match[1])*27.2114)
    return states, energies_eV


def smart_label_placement(energies, labels, state_letter, min_spacing=0.03):
    """
    Умное размещение подписей с автоматическим смещением
    """
    positions = np.array(energies.copy())
    n = len(positions)
    label_texts = [f"{state_letter}$_{ {label} }$ ({energy:.3f} eV)" for label, energy in zip(labels, energies)]
    
    # Сначала сортируем все элементы по энергии
    sorted_indices = np.argsort(positions)
    sorted_positions = positions[sorted_indices]
    sorted_texts = [label_texts[i] for i in sorted_indices]
    
    # Применяем алгоритм смещения
    min_spacing = max(0.03, min_spacing)
    for i in range(1, n):
        if sorted_positions[i] - sorted_positions[i-1] < min_spacing:
            sorted_positions[i] = sorted_positions[i-1] + min_spacing
    
    # Восстанавливаем исходный порядок
    result_positions = np.zeros_like(positions)
    result_texts = [''] * n
    for idx, original_idx in enumerate(sorted_indices):
        result_positions[original_idx] = sorted_positions[idx]
        result_texts[original_idx] = sorted_texts[idx]
    
    return result_positions, result_texts


def plot_singlets_energy_diagram(states, energies_eV):
    """
    Строит диаграмму уровней энергии с интеллектуальным размещением подписей
    """

    global number_of_singlets
    number_of_singlets = len(states)
    
    # Фиксированная длина всех горизонтальных линий
    line_length = 0.5
    
    # Получаем оптимальные позиции для подписей
    energy_range = max(energies_eV) - min(energies_eV)
    adjusted_positions, label_texts = smart_label_placement(energies_eV, states, "S", energy_range*0.05)
    
    # Рисуем горизонтальные линии одинаковой длины
    for i, energy in enumerate(energies_eV):
        # Основная линия уровня
        ax2.hlines(y=energy, xmin=0, xmax=line_length, 
                  color='blue', linewidth=2)
        
        # Подпись в формате "номер (энергия)"
        ax2.text(-0.1, adjusted_positions[i], label_texts[i],
                va='center', ha='right', fontsize=11,
                bbox=dict(facecolor='white', edgecolor='none', pad=2, alpha=0.8))
        
        # Рисуем пунктирный указатель
        ax2.plot([0, -0.09], [energy, adjusted_positions[i]],
                'k:', lw=0.8, alpha=0.5)

    # Настраиваем оси
    ax2.set_ylabel('Energy (eV)', fontsize=12)
    ax2.set_title('Energy Levels Diagram for Excited States', fontsize=14, pad=20)
    ax2.set_xticks([])
    
    # Настраиваем пределы
    ax2.set_xlim(-1, 2.5)
    ax2.set_ylim(min(energies_eV) - 0.1*energy_range, max(energies_eV) + 0.1*energy_range)
    
    # Добавляем сетку
    ax2.grid(axis='y', linestyle=':', alpha=0.4)


def plot_triplets_energy_diagram(states, energies_eV):
    """
    Строит диаграмму уровней энергии с интеллектуальным размещением подписей
    """

    # Фиксированная длина всех горизонтальных линий
    line_length = 0.5
    
    # Получаем оптимальные позиции для подписей
    energy_range = (max(ax2.get_ylim()) - min(ax2.get_ylim()))/1.2
    adjusted_positions, label_texts = smart_label_placement(energies_eV, states, "T", energy_range*0.05)
    
    # Рисуем горизонтальные линии одинаковой длины
    for i, energy in enumerate(energies_eV):
        # Основная линия уровня
        ax2.hlines(y=energy, xmin=1, xmax=1+line_length, 
                  color='red', linewidth=2)
        
        # Подпись в формате "номер (энергия)"
        ax2.text(1.6, adjusted_positions[i], label_texts[i],
                va='center', ha='left', fontsize=11,
                bbox=dict(facecolor='white', edgecolor='none', pad=2, alpha=0.8))
        
        # Рисуем пунктирный указатель
        ax2.plot([1.5, 1.59], [energy, adjusted_positions[i]],
                'k:', lw=0.8, alpha=0.5)
    
    # # Настраиваем пределы
    current_ylim = ax2.get_ylim()
    ylim_min = min(min(current_ylim), min(energies_eV))
    ylim_max = max(max(current_ylim), max(energies_eV))
    energy_range = ylim_max - ylim_min
    ax2.set_ylim(ylim_min - 0.1*energy_range, ylim_max + 0.1*energy_range)


def process_singlet_levels_file():
    try:
        # Чтение данных из файла
        with open(singlet_levels_output_file, 'r') as f:
            text = f.read()
        
        # Извлечение данных и построение диаграммы
        states, energies_eV = extract_energy_levels(text)
        plot_singlets_energy_diagram(states, energies_eV)

        S1_energy = min(energies_eV)
        
    except FileNotFoundError:
        print("Ошибка: файл 'your_file.txt' не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")
    else:
        return S1_energy


def process_triplet_levels_file():
    try:
        # Чтение данных из файла
        with open(triplet_levels_output_file, 'r') as f:
            text = f.read()
        
        # Извлечение данных и построение диаграммы
        states, energies_eV = extract_energy_levels(text)
        states = [state - number_of_singlets for state in states]
        plot_triplets_energy_diagram(states, energies_eV)

        T1_energy = min(energies_eV)
        
    except FileNotFoundError:
        print("Ошибка: файл 'your_file.txt' не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")
    else:
        return T1_energy

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python.exe socme.py <input_file>")
        sys.exit(1)
        
    input_file_name = sys.argv[1]
    input_file_name_cut = os.path.splitext(os.path.basename(input_file_name))[0]

    st = fig.suptitle(input_file_name_cut, fontsize="20")

    extract_fragment(input_file_name, triplet_levels_output_file, start_marker_triplets, end_marker_triplets)
    extract_fragment(input_file_name, singlet_levels_output_file, start_marker_singlets, end_marker_singlets)
    extract_fragment(input_file_name, socme_output_file, start_marker_socme, end_marker_socme)
    
    S1_energy = process_singlet_levels_file()
    T1_energy = process_triplet_levels_file()
    delta_E_ST = S1_energy-T1_energy
    T1_S1_SOCME = process_socme_file()
    T1_S1_SOCME_eV = T1_S1_SOCME*1.23981e-4
    kRISC = T1_S1_SOCME_eV**2 * np.exp(-(delta_E_ST**2))
    
    #ax3.axis('tight')
    ax3.axis('off')

    # Создаем таблицу с основными результатами
    column_label=(
        r"$S1, eV$",
        r"$T1, eV$",
        r"$ΔE_{ST}, eV$",
        r"$|V_{SOC}|, cm^{-1}$",
        r"$|V_{SOC}|^2 ⋅ exp[-(ΔE_{ST})^2]$")


    cellText=[[
        f"{S1_energy:.3f}",
        f"{T1_energy:.3f}",
        f"{delta_E_ST:.3f}",
        f"{T1_S1_SOCME:.3f}",
        f"{kRISC:.2e}"]]

    the_table = ax3.table(cellText=cellText,colLabels=column_label,loc='center',cellLoc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(13)
    the_table.auto_set_column_width([0,1,2,3,4])

    table_cells = the_table.get_celld()
    for cell in table_cells:
        table_cells[cell].PAD = 0.15
        table_cells[cell].set_height(0.4)


    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.1, wspace=0.2)

    fig_filename = input_file_name_cut + "_result.pdf"
    plt.savefig(fig_filename)
    plt.show()

    