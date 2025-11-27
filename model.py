"""
Модель CNN для классификации ЭЭГ данных
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    Архитектура EEGNet для классификации ЭЭГ сигналов
    Основана на статье: Lawhern et al. "EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces"
    """
    
    def __init__(self, n_channels=22, n_timepoints=250, n_classes=4, 
                 F1=8, D=2, F2=16, kernel_length=64, dropout_rate=0.5):
        """
        Инициализация модели
        
        Args:
            n_channels: Количество каналов ЭЭГ
            n_timepoints: Количество временных точек
            n_classes: Количество классов
            F1: Количество фильтров в первом слое
            D: Глубина мультипликатора
            F2: Количество фильтров во втором слое
            kernel_length: Длина ядра временной свертки
            dropout_rate: Коэффициент dropout
        """
        super(EEGNet, self).__init__()
        
        # Первый блок: временная свертка
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), 
                              padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Глубинная свертка (spatial convolution)
        self.conv2 = nn.Conv2d(F1, F1 * D, (n_channels, 1), 
                              groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Второй блок: разделяемая свертка
        self.conv3 = nn.Conv2d(F1 * D, F2, (1, 16), 
                              groups=F1 * D, padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.conv4 = nn.Conv2d(F2, F2, (1, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Вычисление размера после пулинга
        # После pool1: timepoints / 4
        # После pool2: (timepoints / 4) / 8 = timepoints / 32
        self.fc_size = F2 * (n_timepoints // 32)
        
        # Классификатор
        self.fc = nn.Linear(self.fc_size, n_classes)
        
    def forward(self, x):
        """
        Прямой проход
        
        Args:
            x: Входные данные (batch_size, 1, n_channels, n_timepoints)
            
        Returns:
            Выход классификатора
        """
        # Первый блок
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Второй блок
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Выпрямление
        x = x.view(x.size(0), -1)
        
        # Классификация
        x = self.fc(x)
        
        return x


class SimpleCNN(nn.Module):
    """
    Упрощенная CNN архитектура для классификации ЭЭГ
    """
    
    def __init__(self, n_channels=22, n_timepoints=250, n_classes=4):
        """
        Инициализация модели
        
        Args:
            n_channels: Количество каналов ЭЭГ
            n_timepoints: Количество временных точек
            n_classes: Количество классов
        """
        super(SimpleCNN, self).__init__()
        
        # Первый сверточный блок
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Второй сверточный блок
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Третий сверточный блок
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Вычисление размера после пулинга
        # После каждого pool размер уменьшается в 2 раза
        h = n_channels // (2 ** 3)
        w = n_timepoints // (2 ** 3)
        self.fc_size = 128 * h * w
        
        # Полносвязные слои
        self.fc1 = nn.Linear(self.fc_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, n_classes)
        
    def forward(self, x):
        """
        Прямой проход
        
        Args:
            x: Входные данные (batch_size, 1, n_channels, n_timepoints)
            
        Returns:
            Выход классификатора
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

