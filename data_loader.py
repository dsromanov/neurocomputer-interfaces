"""
Модуль для загрузки и предобработки ЭЭГ данных
"""
import numpy as np
import mne
from mne.datasets import sample
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import os


class EEGDataLoader:    
    def __init__(self, sampling_rate=250, n_channels=22, n_samples=1000):
        """
        Инициализация загрузчика данных
        
        Args:
            sampling_rate: Частота дискретизации (Гц)
            n_channels: Количество каналов ЭЭГ
            n_samples: Количество образцов на класс
        """
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.scaler = StandardScaler()
        
    def generate_synthetic_eeg_data(self, n_classes=4):
        """
        Генерация синтетических ЭЭГ данных на основе реальных паттернов
        
        Args:
            n_classes: Количество классов (движений/воображений движений)
            
        Returns:
            X: Массив данных (n_samples * n_classes, n_channels, time_points)
            y: Метки классов
        """
        np.random.seed(42)
        n_time_points = int(self.sampling_rate * 4)  # 4 секунды данных
        
        X = []
        y = []
        
        for class_id in range(n_classes):
            for _ in range(self.n_samples):
                # Генерируем базовый сигнал с разными частотными характеристиками для каждого класса
                t = np.linspace(0, 4, n_time_points)
                
                # Разные частотные диапазоны для разных классов
                # Класс 0: альфа-ритм (8-13 Гц) - расслабление
                # Класс 1: бета-ритм (13-30 Гц) - активное мышление
                # Класс 2: гамма-ритм (30-100 Гц) - когнитивная обработка
                # Класс 3: тета-ритм (4-8 Гц) - медитация/сон
                
                freq_ranges = {
                    0: (8, 13),   # Альфа
                    1: (13, 30),  # Бета
                    2: (30, 50),  # Гамма (низкая)
                    3: (4, 8)     # Тета
                }
                
                freq_min, freq_max = freq_ranges[class_id]
                center_freq = (freq_min + freq_max) / 2
                
                # Генерируем сигнал для каждого канала
                channel_data = np.zeros((self.n_channels, n_time_points))
                
                for ch in range(self.n_channels):
                    # Основной ритм
                    signal_component = np.sin(2 * np.pi * center_freq * t)
                    
                    # Добавляем шум и вариативность
                    noise = np.random.normal(0, 0.3, n_time_points)
                    amplitude_modulation = 1 + 0.2 * np.sin(2 * np.pi * 0.5 * t)
                    
                    # Разные каналы имеют разные амплитуды
                    channel_amplitude = 0.5 + 0.3 * np.random.rand()
                    
                    channel_data[ch] = channel_amplitude * signal_component * amplitude_modulation + noise
                    
                    # Добавляем артефакты (глазные движения, мышечные)
                    if np.random.rand() > 0.7:
                        artifact = np.random.normal(0, 0.5, n_time_points)
                        channel_data[ch] += artifact
                
                X.append(channel_data)
                y.append(class_id)
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y
    
    def preprocess_data(self, X, apply_filter=True, apply_ica=False):
        """
        Предобработка ЭЭГ данных
        
        Args:
            X: Массив данных (n_samples, n_channels, time_points)
            apply_filter: Применить фильтрацию
            apply_ica: Применить ICA (пока не реализовано)
            
        Returns:
            X_processed: Предобработанные данные
        """
        X_processed = X.copy()
        
        if apply_filter:
            # Применяем bandpass фильтр (1-50 Гц) для удаления дрейфа и высокочастотного шума
            for i in range(X_processed.shape[0]):
                for ch in range(X_processed.shape[1]):
                    # Bandpass фильтр
                    sos = signal.butter(4, [1, 50], btype='band', 
                                       fs=self.sampling_rate, output='sos')
                    X_processed[i, ch, :] = signal.sosfilt(sos, X_processed[i, ch, :])
        
        # Нормализация по каналам
        n_samples, n_channels, n_time = X_processed.shape
        X_reshaped = X_processed.reshape(-1, n_time)
        X_normalized = self.scaler.fit_transform(X_reshaped.T).T
        X_processed = X_normalized.reshape(n_samples, n_channels, n_time)
        
        return X_processed
    
    def create_windows(self, X, y, window_size=250, overlap=0.5):
        """
        Создание окон из непрерывных данных
        
        Args:
            X: Массив данных
            y: Метки
            window_size: Размер окна в сэмплах
            overlap: Перекрытие окон (0-1)
            
        Returns:
            X_windows: Данные в виде окон
            y_windows: Метки для окон
        """
        X_windows = []
        y_windows = []
        
        step = int(window_size * (1 - overlap))
        
        for i in range(len(X)):
            n_time = X[i].shape[1]
            for start in range(0, n_time - window_size + 1, step):
                window = X[i][:, start:start + window_size]
                X_windows.append(window)
                y_windows.append(y[i])
        
        return np.array(X_windows), np.array(y_windows)
    
    def load_mne_sample_data(self):
        """
        Загрузка ЭЭГ данных из MNE sample dataset (Motor Imagery)
        
        Returns:
            X: Массив данных (n_samples, n_channels, time_points)
            y: Метки классов
        """
        print("Загрузка MNE sample dataset...")
        print("Это может занять некоторое время при первом запуске...")
        
        data_path = sample.data_path()
        
        raw_fname = os.path.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
        raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
        
        raw.pick('eeg', exclude='bads')
        
        data, times = raw[:, :]
        
        self.sampling_rate = int(raw.info['sfreq'])
        self.n_channels = data.shape[0]
        
        print(f"Загружено: {self.n_channels} каналов, частота дискретизации: {self.sampling_rate} Гц")
        print(f"Длительность записи: {times[-1]:.1f} секунд")
        
        segment_length = int(self.sampling_rate * 4)  # 4 секунды
        n_segments = data.shape[1] // segment_length
        
        X = []
        y = []
        
        for i in range(n_segments):
            start_idx = i * segment_length
            end_idx = start_idx + segment_length
            segment = data[:, start_idx:end_idx]
            
            freqs, psd = signal.welch(segment, fs=self.sampling_rate, nperseg=min(256, segment_length))
            
            # Альфа (8-13 Гц)
            alpha_mask = (freqs >= 8) & (freqs <= 13)
            alpha_power = np.mean(psd[:, alpha_mask], axis=1).mean()
            
            # Бета (13-30 Гц)
            beta_mask = (freqs >= 13) & (freqs <= 30)
            beta_power = np.mean(psd[:, beta_mask], axis=1).mean()
            
            # Гамма (30-50 Гц)
            gamma_mask = (freqs >= 30) & (freqs <= 50)
            gamma_power = np.mean(psd[:, gamma_mask], axis=1).mean()
            
            # Тета (4-8 Гц)
            theta_mask = (freqs >= 4) & (freqs <= 8)
            theta_power = np.mean(psd[:, theta_mask], axis=1).mean()
            
            powers = [alpha_power, beta_power, gamma_power, theta_power]
            max_power = max(powers)
            
            threshold = max_power * 0.7  # Порог 70% от максимума
            
            valid_classes = [i for i, p in enumerate(powers) if p >= threshold]
            
            if len(valid_classes) > 0:
                dominant_class = np.argmax(powers)
            else:
                dominant_class = np.argmax(powers)
            
            X.append(segment)
            y.append(dominant_class)
        
        X = np.array(X)
        y = np.array(y)
        
        # Нормализуем метки в диапазон [0, n_classes-1]
        unique_labels = np.unique(y)
        n_classes = len(unique_labels)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        y = np.array([label_mapping[label] for label in y])
        
        print(f"Создано {len(X)} сегментов")
        print(f"Уникальные классы: {unique_labels} -> нормализованы в [0, {n_classes-1}]")
        print(f"Распределение по классам (до балансировки): {np.bincount(y)}")
        
        return X, y
    
    def balance_classes(self, X, y, method='undersample', random_state=42):
        """
        Балансировка классов в данных
        
        Args:
            X: Массив данных
            y: Метки классов
            method: Метод балансировки ('undersample', 'oversample', 'smote')
            random_state: Seed для воспроизводимости
            
        Returns:
            X_balanced, y_balanced: Сбалансированные данные
        """
        np.random.seed(random_state)
        
        unique_labels = np.unique(y)
        class_counts = np.bincount(y)
        min_samples = class_counts[class_counts > 0].min()
        max_samples = class_counts.max()
        
        print(f"\nБалансировка классов (метод: {method}):")
        print(f"  Минимум образцов в классе: {min_samples}")
        print(f"  Максимум образцов в классе: {max_samples}")
        
        if method == 'undersample':
            X_balanced = []
            y_balanced = []
            
            for label in unique_labels:
                indices = np.where(y == label)[0]
                if len(indices) > 0:
                    target_samples = max(min_samples, len(indices) // 2)
                    n_samples = min(target_samples, len(indices))
                    selected_indices = np.random.choice(indices, size=n_samples, replace=False)
                    X_balanced.append(X[selected_indices])
                    y_balanced.append(y[selected_indices])
            
            X_balanced = np.concatenate(X_balanced, axis=0)
            y_balanced = np.concatenate(y_balanced, axis=0)
            
        elif method == 'oversample':
            # Oversampling: увеличиваем минорные классы до среднего значения
            avg_samples = int(class_counts[class_counts > 0].mean())
            target_samples = max(avg_samples, min_samples * 2)  # Не менее чем в 2 раза больше минимума
            
            X_balanced = []
            y_balanced = []
            
            for label in unique_labels:
                indices = np.where(y == label)[0]
                if len(indices) > 0:
                    n_samples = max(target_samples, len(indices))
                    if len(indices) < n_samples:
                        selected_indices = resample(indices, n_samples=n_samples, 
                                                  random_state=random_state, replace=True)
                    else:
                        selected_indices = np.random.choice(indices, size=n_samples, replace=False)
                    X_balanced.append(X[selected_indices])
                    y_balanced.append(y[selected_indices])
            
            X_balanced = np.concatenate(X_balanced, axis=0)
            y_balanced = np.concatenate(y_balanced, axis=0)
            
        elif method == 'balanced':
            # Сбалансированный подход: берем среднее количество из каждого класса
            avg_samples = int(class_counts[class_counts > 0].mean())
            
            X_balanced = []
            y_balanced = []
            
            for label in unique_labels:
                indices = np.where(y == label)[0]
                if len(indices) > 0:
                    n_samples = min(avg_samples, len(indices))
                    if len(indices) < n_samples:
                        # Если меньше среднего, делаем oversample
                        selected_indices = resample(indices, n_samples=n_samples, 
                                                  random_state=random_state, replace=True)
                    else:
                        # Если больше среднего, делаем undersample
                        selected_indices = np.random.choice(indices, size=n_samples, replace=False)
                    X_balanced.append(X[selected_indices])
                    y_balanced.append(y[selected_indices])
            
            X_balanced = np.concatenate(X_balanced, axis=0)
            y_balanced = np.concatenate(y_balanced, axis=0)
            
        else:
            X_balanced = X
            y_balanced = y
        
        shuffle_indices = np.random.permutation(len(X_balanced))
        X_balanced = X_balanced[shuffle_indices]
        y_balanced = y_balanced[shuffle_indices]
        
        print(f"  После балансировки: {len(X_balanced)} образцов")
        print(f"  Распределение по классам: {np.bincount(y_balanced)}")
        
        return X_balanced, y_balanced
    
    def load_physionet_mi_data(self, subjects=[1], runs=[3, 7, 11]):
        """
        Загрузка данных Motor Imagery из PhysioNet BCI Competition
        
        Примечание: Для использования этого метода нужно скачать данные вручную
        с https://www.physionet.org/content/eegmmidb/1.0.0/
        
        Args:
            subjects: Список номеров субъектов (1-109)
            runs: Список номеров сессий (3, 7, 11 - это воображение движений)
        
        Returns:
            X: Массив данных
            y: Метки классов
        """
        print("Загрузка PhysioNet Motor Imagery данных...")
        print("Убедитесь, что данные скачаны в папку 'data/physionet/'")
        
        data_dir = 'data/physionet/'
        
        if not os.path.exists(data_dir):
            print(f"Ошибка: Папка {data_dir} не найдена!")
            print("Пожалуйста, скачайте данные с https://www.physionet.org/content/eegmmidb/1.0.0/")
            raise FileNotFoundError(f"Папка {data_dir} не найдена")
        
        X_all = []
        y_all = []
        
        # Классы: 0 - левая рука, 1 - правая рука, 2 - обе ноги, 3 - язык
        class_mapping = {1: 0, 2: 1, 3: 2, 4: 3}
        
        for subject in subjects:
            for run in runs:
                # Формат имени файла: S001R03.edf
                fname = f'S{subject:03d}R{run:02d}.edf'
                filepath = os.path.join(data_dir, fname)
                
                if os.path.exists(filepath):
                    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
                    raw.pick('eeg', exclude='bads')
                    
                    # Получаем события из аннотаций
                    events, event_dict = mne.events_from_annotations(raw, verbose=False)
                    
                    # Извлекаем сегменты для каждого события
                    for event in events:
                        event_id = event[2]
                        if event_id in class_mapping:
                            start_time = event[0] / raw.info['sfreq']
                            end_time = start_time + 4  # 4 секунды
                            
                            # Извлекаем сегмент
                            raw_segment = raw.copy().crop(tmin=start_time, tmax=end_time)
                            data, _ = raw_segment[:, :]
                            
                            X_all.append(data)
                            y_all.append(class_mapping[event_id])
        
        if len(X_all) == 0:
            raise ValueError("Не найдено данных. Проверьте путь к данным.")
        
        X = np.array(X_all)
        y = np.array(y_all)
        
        # Нормализуем метки в диапазон [0, n_classes-1]
        unique_labels = np.unique(y)
        n_classes = len(unique_labels)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        y = np.array([label_mapping[label] for label in y])
        
        self.sampling_rate = int(raw.info['sfreq'])
        self.n_channels = X.shape[1]
        
        print(f"Загружено {len(X)} сегментов")
        print(f"Уникальные классы: {unique_labels} -> нормализованы в [0, {n_classes-1}]")
        print(f"Распределение по классам: {np.bincount(y)}")
        
        return X, y
    
    def load_and_prepare_data(self, use_real_data=True, test_size=0.2, val_size=0.1,
                             balance_classes=True, balance_method='undersample'):
        """
        Полная загрузка и подготовка данных
        
        Args:
            use_real_data: Если True, использует реальные данные из MNE sample dataset,
                          иначе использует синтетические данные
            test_size: Доля тестовой выборки
            val_size: Доля валидационной выборки
            balance_classes: Если True, применяет балансировку классов
            balance_method: Метод балансировки ('undersample', 'oversample')
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        if use_real_data:
            print("Использование реальных данных из MNE sample dataset...")
            try:
                X, y = self.load_mne_sample_data()
            except Exception as e:
                print(f"Ошибка при загрузке реальных данных: {e}")
                print("Переключение на синтетические данные...")
                X, y = self.generate_synthetic_eeg_data(n_classes=4)
        else:
            print("Генерация синтетических ЭЭГ данных...")
            X, y = self.generate_synthetic_eeg_data(n_classes=4)
        
        print("Предобработка данных...")
        X = self.preprocess_data(X, apply_filter=True)
        
        print("Создание окон...")
        window_size = int(self.sampling_rate * 1)  # 1 секунда
        X, y = self.create_windows(X, y, window_size=window_size, overlap=0.5)
        
        if balance_classes:
            X, y = self.balance_classes(X, y, method=balance_method, random_state=42)
        
        unique_labels = np.unique(y)
        n_classes = len(unique_labels)
        if n_classes == 0:
            raise ValueError("Не найдено ни одного класса в данных!")
        
        # Нормализуем метки в диапазон [0, n_classes-1]
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        y = np.array([label_mapping[label] for label in y])
        
        print(f"После создания окон: {len(X)} образцов, {n_classes} классов")
        print(f"Распределение по классам: {np.bincount(y)}")
        
        min_samples_per_class = np.bincount(y).min()
        if min_samples_per_class < 2:
            print("Предупреждение: некоторые классы имеют менее 2 образцов, используем обычное разделение")
            stratify = None
        else:
            stratify = y
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + val_size), random_state=42, stratify=stratify
        )
        
        # Проверяем стратификацию для валидации/теста
        min_samples_val = np.bincount(y_temp).min() if len(np.unique(y_temp)) > 0 else 0
        stratify_val = y_temp if min_samples_val >= 2 else None
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_size/(test_size + val_size), 
            random_state=42, stratify=stratify_val
        )
        
        all_labels = np.concatenate([y_train, y_val, y_test])
        min_label = all_labels.min()
        max_label = all_labels.max()
        n_unique = len(np.unique(all_labels))
        
        if min_label < 0 or max_label >= n_unique:
            raise ValueError(f"Метки выходят за допустимый диапазон: min={min_label}, max={max_label}, n_classes={n_unique}")
        
        print(f"Размер обучающей выборки: {X_train.shape}")
        print(f"Размер валидационной выборки: {X_val.shape}")
        print(f"Размер тестовой выборки: {X_test.shape}")
        print(f"Метки в диапазоне: [{min_label}, {max_label}], уникальных классов: {n_unique}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

