"""
Скрипт для обработки аудиофайлов с помощью модели Whisper
Загружает аудио по URL из файла urls_normalized.tsv и применяет модель для распознавания речи
"""

import os
import requests
import torch
import torchaudio
import pandas as pd
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from typing import List, Dict, Optional
import logging
from tqdm import tqdm
import time
import random

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioProcessor:
    """Класс для обработки аудиофайлов с помощью Whisper"""
    
    def __init__(self, model_name: str = "openai/whisper-small"):
        """
        Инициализация процессора аудио
        
        Args:
            model_name: Название модели Whisper для использования
        """
        self.model_name = model_name
        self.processor = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Загрузка модели и процессора Whisper"""
        try:
            logger.info(f"Загружаем модель {self.model_name}...")
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
            self.model.config.forced_decoder_ids = None
            logger.info("Модель успешно загружена!")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise
    
    def download_audio(self, url: str, save_path: str) -> bool:
        """
        Загрузка аудиофайла по URL
        
        Args:
            url: URL аудиофайла
            save_path: Путь для сохранения файла
            
        Returns:
            bool: True если загрузка успешна, False иначе
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Аудиофайл загружен: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке {url}: {e}")
            return False
    
    def load_audio(self, file_path: str, target_sr: int = 16000) -> Optional[Dict]:
        """
        Загрузка и предобработка аудиофайла
        
        Args:
            file_path: Путь к аудиофайлу
            target_sr: Целевая частота дискретизации
            
        Returns:
            Dict с данными аудио или None при ошибке
        """
        try:
            # Загружаем аудио
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Конвертируем в моно если нужно
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Ресэмплируем если нужно
            if sample_rate != target_sr:
                waveform = torchaudio.functional.resample(
                    waveform, orig_freq=sample_rate, new_freq=target_sr
                )
                sample_rate = target_sr
            
            # Конвертируем в numpy array
            audio_array = waveform.squeeze().numpy()
            
            return {
                'array': audio_array,
                'sampling_rate': sample_rate,
                'file_path': file_path
            }
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке аудио {file_path}: {e}")
            return None
    
    def transcribe_audio(self, audio_data: Dict) -> Optional[str]:
        """
        Транскрипция аудио с помощью Whisper
        
        Args:
            audio_data: Данные аудио
            
        Returns:
            str: Транскрибированный текст или None при ошибке
        """
        try:
            # Подготавливаем входные данные
            input_features = self.processor(
                audio_data['array'], 
                sampling_rate=audio_data['sampling_rate'], 
                return_tensors="pt"
            ).input_features
            
            # Генерируем предсказания
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features)
            
            # Декодируем в текст
            transcription = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
            
            return transcription
            
        except Exception as e:
            logger.error(f"Ошибка при транскрипции: {e}")
            return None
    
    def process_audio_from_url(self, url: str, temp_dir: str = "temp_audio") -> Optional[str]:
        """
        Полный цикл обработки аудио от URL до транскрипции
        
        Args:
            url: URL аудиофайла
            temp_dir: Временная директория для сохранения файлов
            
        Returns:
            str: Транскрибированный текст или None при ошибке
        """
        # Создаем временную директорию если не существует
        os.makedirs(temp_dir, exist_ok=True)
        
        # Генерируем имя файла
        filename = url.split('/')[-1] + '.wav'
        file_path = os.path.join(temp_dir, filename)
        
        try:
            # Загружаем аудио
            if not self.download_audio(url, file_path):
                return None
            
            # Загружаем и предобрабатываем аудио
            audio_data = self.load_audio(file_path)
            if audio_data is None:
                return None
            
            # Транскрибируем
            transcription = self.transcribe_audio(audio_data)
            
            # Удаляем временный файл
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return transcription
            
        except Exception as e:
            logger.error(f"Ошибка при обработке {url}: {e}")
            # Удаляем временный файл при ошибке
            if os.path.exists(file_path):
                os.remove(file_path)
            return None

def load_urls_from_file(file_path: str) -> List[str]:
    """
    Загрузка URL из файла
    
    Args:
        file_path: Путь к файлу с URL
        
    Returns:
        List[str]: Список URL
    """
    urls = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and line.startswith('http'):
                    urls.append(line)
        
        logger.info(f"Загружено {len(urls)} URL из файла {file_path}")
        return urls
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке URL из файла {file_path}: {e}")
        return []

def create_prediction_dataset(urls: List[str], processor: AudioProcessor, 
                            num_samples: int = 10, random_seed: int = 42) -> pd.DataFrame:
    """
    Создание датасета с предсказаниями модели
    
    Args:
        urls: Список URL аудиофайлов
        processor: Процессор аудио
        num_samples: Количество случайных образцов для обработки
        random_seed: Seed для воспроизводимости
        
    Returns:
        pd.DataFrame: Датафрейм с результатами
    """
    # Устанавливаем seed для воспроизводимости
    random.seed(random_seed)
    
    # Выбираем случайные URL
    selected_urls = random.sample(urls, min(num_samples, len(urls)))
    
    results = []
    
    logger.info(f"Обрабатываем {len(selected_urls)} аудиофайлов...")
    
    for i, url in enumerate(tqdm(selected_urls, desc="Обработка аудио")):
        logger.info(f"Обрабатываем файл {i+1}/{len(selected_urls)}: {url}")
        
        # Транскрибируем аудио
        transcription = processor.process_audio_from_url(url)
        
        results.append({
            'url': url,
            'transcription': transcription if transcription else "Ошибка транскрипции",
            'success': transcription is not None
        })
        
        # Небольшая пауза между запросами
        time.sleep(0.5)
    
    return pd.DataFrame(results)

def main():
    """Основная функция для демонстрации работы"""
    
    # Путь к файлу с URL
    urls_file = "urls_normalized.tsv"
    
    # Проверяем существование файла
    if not os.path.exists(urls_file):
        logger.error(f"Файл {urls_file} не найден!")
        return
    
    # Загружаем URL
    urls = load_urls_from_file(urls_file)
    if not urls:
        logger.error("Не удалось загрузить URL из файла!")
        return
    
    # Создаем процессор
    processor = AudioProcessor()
    
    # Создаем датасет с предсказаниями
    results_df = create_prediction_dataset(urls, processor, num_samples=10)
    
    # Выводим результаты
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ ТРАНСКРИПЦИИ АУДИОФАЙЛОВ")
    print("="*80)
    
    for idx, row in results_df.iterrows():
        print(f"\nФайл {idx + 1}:")
        print(f"URL: {row['url']}")
        print(f"Транскрипция: {row['transcription']}")
        print(f"Статус: {'Успешно' if row['success'] else 'Ошибка'}")
        print("-" * 80)
    
    # Сохраняем результаты
    output_file = "transcription_results.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"Результаты сохранены в файл: {output_file}")
    
    # Статистика
    successful = results_df['success'].sum()
    total = len(results_df)
    print(f"\nСтатистика:")
    print(f"Успешно обработано: {successful}/{total}")
    print(f"Процент успеха: {successful/total*100:.1f}%")

if __name__ == "__main__":
    main()

