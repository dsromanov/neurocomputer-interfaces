import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import os

from data_loader import EEGDataLoader
from model import EEGNet, SimpleCNN


class EEGDataset(Dataset):
    """Класс для работы с данными ЭЭГ в PyTorch"""
    
    def __init__(self, X, y):
        """
        Args:
            X: Массив данных (n_samples, n_channels, time_points)
            y: Метки классов
        """
        self.X = torch.FloatTensor(X)
        y = np.array(y)
        
        unique_labels = np.unique(y)
        min_label = y.min()
        max_label = y.max()
        n_classes = len(unique_labels)
        
        if min_label < 0 or max_label >= n_classes:
            # Нормализуем метки
            label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
            y = np.array([label_mapping[label] for label in y])
            print(f"Предупреждение: Метки нормализованы в диапазон [0, {n_classes-1}]")
        
        self.y = torch.LongTensor(y)
        
        assert self.y.min() >= 0, f"Минимальная метка должна быть >= 0, получено {self.y.min()}"
        assert self.y.max() < n_classes, f"Максимальная метка должна быть < {n_classes}, получено {self.y.max()}"
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx].unsqueeze(0)  # (1, n_channels, time_points)
        return x, self.y[idx]


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Обучение модели на одной эпохе"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Валидация модели"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path='results/training_history.png'):
    """Визуализация истории обучения"""
    os.makedirs('results', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # График потерь
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Val Loss', marker='s')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Loss')
    ax1.set_title('История обучения: Loss')
    ax1.legend()
    ax1.grid(True)
    
    # График точности
    ax2.plot(train_accs, label='Train Accuracy', marker='o')
    ax2.plot(val_accs, label='Val Accuracy', marker='s')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('История обучения: Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"График сохранен: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='results/confusion_matrix.png'):
    """Визуализация матрицы ошибок"""
    os.makedirs('results', exist_ok=True)
    
    all_labels = np.concatenate([y_true, y_pred])
    unique_labels = np.unique(all_labels)
    n_classes = len(unique_labels)
    
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    used_class_names = [class_names[i] if i < len(class_names) else f'Класс {i}' 
                        for i in unique_labels]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=used_class_names, yticklabels=used_class_names)
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.title('Матрица ошибок')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Матрица ошибок сохранена: {save_path}")
    plt.close()


def main():
    """Основная функция обучения"""
    batch_size = 32
    n_epochs = 50
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Используемое устройство: {device}")
    print("=" * 50)
    
    data_loader = EEGDataLoader(sampling_rate=250, n_channels=22, n_samples=500)
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.load_and_prepare_data(
        use_real_data=True, balance_classes=True, balance_method='balanced'
    )
    
    print(f"\nПроверка меток перед созданием датасетов:")
    print(f"  Train: min={y_train.min()}, max={y_train.max()}, unique={np.unique(y_train)}")
    print(f"  Val: min={y_val.min()}, max={y_val.max()}, unique={np.unique(y_val)}")
    print(f"  Test: min={y_test.min()}, max={y_test.max()}, unique={np.unique(y_test)}")
    
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    test_dataset = EEGDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    n_channels = X_train.shape[1]
    n_timepoints = X_train.shape[2]
    
    all_labels = np.concatenate([y_train, y_val, y_test])
    unique_labels = np.unique(all_labels)
    n_classes = len(unique_labels)
    
    # Проверяем диапазон меток
    min_label = all_labels.min()
    max_label = all_labels.max()
    
    print(f"\nПараметры модели:")
    print(f"  Каналы: {n_channels}")
    print(f"  Временные точки: {n_timepoints}")
    print(f"  Классы: {n_classes}")
    print(f"  Диапазон меток: [{min_label}, {max_label}]")
    print(f"  Уникальные метки: {unique_labels}")
    
    if min_label < 0 or max_label >= n_classes:
        raise ValueError(f"Метки выходят за допустимый диапазон! min={min_label}, max={max_label}, n_classes={n_classes}")
    
    model = EEGNet(n_channels=n_channels, n_timepoints=n_timepoints, n_classes=n_classes)
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nВсего параметров: {total_params:,}")
    print(f"Обучаемых параметров: {trainable_params:,}")
    
    # Функция потерь с весами классов для учета дисбаланса
    # Вычисляем веса обратно пропорциональные частоте классов
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights = total_samples / (len(class_counts) * class_counts.astype(float))
    class_weights = class_weights / class_weights.sum() * len(class_counts)  # Нормализуем
    
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    print(f"\nВеса классов для функции потерь: {class_weights}")
    print(f"  Распределение классов в обучающей выборке: {class_counts}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=5, verbose=True)
    
    print("\n" + "=" * 50)
    print("Начало обучения...")
    print("=" * 50)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    
    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        scheduler.step(val_loss)
        
        print(f"Эпоха {epoch+1}/{n_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('results', exist_ok=True)
            torch.save(model.state_dict(), 'results/best_model.pth')
            print(f"  ✓ Сохранена лучшая модель (Val Acc: {val_acc:.4f})")
        print()
    
    model.load_state_dict(torch.load('results/best_model.pth'))
    
    print("=" * 50)
    print("Тестирование на тестовой выборке...")
    print("=" * 50)
    
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)
    
    print(f"\nРезультаты на тестовой выборке:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    
    class_name_map = {
        0: 'Класс 0 (Левая рука)',
        1: 'Класс 1 (Правая рука)',
        2: 'Класс 2 (Левая нога)',
        3: 'Класс 3 (Правая нога)'
    }
    unique_labels = np.unique(test_labels)
    class_names = [class_name_map.get(i, f'Класс {i}') for i in range(n_classes)]
    
    print("\n" + "=" * 50)
    print("Детальный отчет по классам:")
    print("=" * 50)
    print(classification_report(test_labels, test_preds, target_names=class_names, 
                                labels=unique_labels, zero_division=0))
    
    print("\n" + "=" * 50)
    print("Информация о распределении классов:")
    print("=" * 50)
    for i, label in enumerate(unique_labels):
        count = np.sum(test_labels == label)
        pred_count = np.sum(test_preds == label)
        print(f"  {class_names[i]}: {count} образцов в тесте, {pred_count} предсказано")
    
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    plot_confusion_matrix(test_labels, test_preds, class_names)
    
    print("\n" + "=" * 50)
    print("Обучение завершено!")
    print("=" * 50)


if __name__ == "__main__":
    main()

