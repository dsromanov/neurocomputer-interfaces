"""
Скрипт для визуализации ЭЭГ данных
"""
import numpy as np
import matplotlib.pyplot as plt
from data_loader import EEGDataLoader
import os


def visualize_eeg_signals(X, y, n_examples=4, save_path='results/eeg_signals.png'):
    """
    Визуализация примеров ЭЭГ сигналов
    
    Args:
        X: Массив данных (n_samples, n_channels, time_points)
        y: Метки классов
        n_examples: Количество примеров для каждого класса
        save_path: Путь для сохранения
    """
    os.makedirs('results', exist_ok=True)
    
    n_classes = len(np.unique(y))
    n_channels = X.shape[1]
    time_points = X.shape[2]
    
    channels_to_show = min(5, n_channels)
    selected_channels = np.linspace(0, n_channels-1, channels_to_show, dtype=int)
    
    fig, axes = plt.subplots(n_classes, 1, figsize=(14, 3 * n_classes))
    if n_classes == 1:
        axes = [axes]
    
    class_names = ['Левая рука', 'Правая рука', 'Левая нога', 'Правая нога']
    
    for class_id in range(n_classes):
        class_indices = np.where(y == class_id)[0]
        if len(class_indices) == 0:
            continue
        
        example_idx = class_indices[0]
        example_data = X[example_idx]
        
        time_axis = np.linspace(0, 4, time_points)  # 4 секунды
        
        for i, ch_idx in enumerate(selected_channels):
            offset = i * 3
            axes[class_id].plot(time_axis, example_data[ch_idx] + offset, 
                              label=f'Канал {ch_idx}', linewidth=1.5)
        
        axes[class_id].set_xlabel('Время (сек)')
        axes[class_id].set_ylabel('Амплитуда (мкВ)')
        class_name = class_names[class_id] if class_id < len(class_names) else f'Класс {class_id}'
        axes[class_id].set_title(f'Класс {class_id}: {class_name}')
        axes[class_id].legend(loc='upper right', fontsize=8)
        axes[class_id].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Визуализация сохранена: {save_path}")
    plt.close()


def visualize_spectrogram(X, y, n_examples=1, save_path='results/spectrograms.png', sampling_rate=250):
    """
    Визуализация спектрограмм ЭЭГ сигналов
    
    Args:
        X: Массив данных
        y: Метки классов
        n_examples: Количество примеров для каждого класса
        save_path: Путь для сохранения
        sampling_rate: Частота дискретизации (Гц)
    """
    from scipy import signal
    
    os.makedirs('results', exist_ok=True)
    
    n_classes = len(np.unique(y))
    
    fig, axes = plt.subplots(n_classes, 1, figsize=(14, 3 * n_classes))
    if n_classes == 1:
        axes = [axes]
    
    class_names = ['Левая рука', 'Правая рука', 'Левая нога', 'Правая нога']
    
    for class_id in range(n_classes):
        class_indices = np.where(y == class_id)[0]
        if len(class_indices) == 0:
            continue
        
        example_idx = class_indices[0]
        example_data = X[example_idx]
        
        ch_idx = example_data.shape[0] // 2
        signal_data = example_data[ch_idx]
        
        frequencies, times, Sxx = signal.spectrogram(
            signal_data, sampling_rate, nperseg=64, noverlap=32
        )
        
        freq_mask = frequencies <= 50
        frequencies = frequencies[freq_mask]
        Sxx = Sxx[freq_mask, :]
        
        im = axes[class_id].pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10), 
                                      shading='gouraud', cmap='viridis')
        axes[class_id].set_xlabel('Время (сек)')
        axes[class_id].set_ylabel('Частота (Гц)')
        class_name = class_names[class_id] if class_id < len(class_names) else f'Класс {class_id}'
        axes[class_id].set_title(f'Спектрограмма - Класс {class_id}: {class_name}')
        plt.colorbar(im, ax=axes[class_id], label='Мощность (дБ)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Спектрограммы сохранены: {save_path}")
    plt.close()


def main():
    print("Загрузка данных для визуализации...")
    
    data_loader = EEGDataLoader(sampling_rate=250, n_channels=22, n_samples=100)
    try:
        X, y = data_loader.load_mne_sample_data()
        sampling_rate = data_loader.sampling_rate
    except Exception as e:
        print(f"Ошибка при загрузке реальных данных: {e}")
        print("Использование синтетических данных...")
        X, y = data_loader.generate_synthetic_eeg_data(n_classes=4)
        sampling_rate = data_loader.sampling_rate
    
    X = data_loader.preprocess_data(X, apply_filter=True)
    
    print("Визуализация ЭЭГ сигналов...")
    visualize_eeg_signals(X, y)
    
    print("Визуализация спектрограмм...")
    visualize_spectrogram(X, y, sampling_rate=sampling_rate)
    
    print("\nВизуализация завершена!")


if __name__ == "__main__":
    main()

