"""
Скрипт для обработки локальных аудиофайлов с помощью Whisper
Работает с файлами, скачанными в папку audio_files
"""

import os
import torch
import torchaudio
import pandas as pd
import random
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm
import glob

class LocalAudioProcessor:
    """Класс для обработки локальных аудиофайлов"""
    
    def __init__(self, model_name="openai/whisper-small"):
        self.model_name = model_name
        self.processor = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Загрузка модели Whisper"""
        print(f"Загружаем модель {self.model_name}...")
        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
        self.model.config.forced_decoder_ids = None
        print("Модель загружена!")
    
    def load_audio_file(self, file_path, target_sr=16000):
        """Загрузка и предобработка аудиофайла"""
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
            print(f"Ошибка при загрузке {file_path}: {e}")
            return None
    
    def transcribe_audio(self, audio_data):
        """Транскрипция аудио"""
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
            print(f"Ошибка при транскрипции: {e}")
            return None
    
    def process_audio_file(self, file_path):
        """Полный цикл обработки аудиофайла"""
        # Загружаем аудио
        audio_data = self.load_audio_file(file_path)
        if audio_data is None:
            return None
        
        # Транскрибируем
        transcription = self.transcribe_audio(audio_data)
        return transcription

def get_audio_files_from_folder(folder_path, extensions=None):
    """Получение списка аудиофайлов из папки"""
    if extensions is None:
        extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg']
    
    audio_files = []
    for ext in extensions:
        pattern = os.path.join(folder_path, ext)
        audio_files.extend(glob.glob(pattern))
    
    return sorted(audio_files)

def process_local_audio_files(folder_path="audio_files", num_samples=10, random_seed=42):
    """
    Обработка локальных аудиофайлов
    
    Args:
        folder_path: Папка с аудиофайлами
        num_samples: Количество файлов для обработки
        random_seed: Seed для воспроизводимости
    """
    
    # Проверяем существование папки
    if not os.path.exists(folder_path):
        print(f"Папка {folder_path} не найдена!")
        return pd.DataFrame()
    
    # Получаем список аудиофайлов
    audio_files = get_audio_files_from_folder(folder_path)
    
    if not audio_files:
        print(f"В папке {folder_path} не найдено аудиофайлов!")
        return pd.DataFrame()
    
    print(f"Найдено {len(audio_files)} аудиофайлов в папке {folder_path}")
    
    # Выбираем случайные файлы
    random.seed(random_seed)
    selected_files = random.sample(audio_files, min(num_samples, len(audio_files)))
    
    # Создаем процессор
    processor = LocalAudioProcessor()
    
    # Обрабатываем файлы
    results = []
    print(f"\nОбрабатываем {len(selected_files)} файлов...")
    
    for i, file_path in enumerate(tqdm(selected_files, desc="Обработка")):
        filename = os.path.basename(file_path)
        print(f"\n[{i+1}/{len(selected_files)}] Обрабатываем: {filename}")
        
        transcription = processor.process_audio_file(file_path)
        
        results.append({
            'filename': filename,
            'file_path': file_path,
            'transcription': transcription if transcription else "Ошибка транскрипции",
            'success': transcription is not None
        })
    
    return pd.DataFrame(results)

def main():
    """Основная функция"""
    print("🎵 Обработка локальных аудиофайлов с помощью Whisper")
    print("=" * 60)
    
    # Обрабатываем файлы
    results_df = process_local_audio_files(
        folder_path="audio_files",
        num_samples=10
    )
    
    if results_df.empty:
        print("Нет данных для обработки!")
        return
    
    # Выводим результаты
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ ТРАНСКРИПЦИИ ЛОКАЛЬНЫХ АУДИОФАЙЛОВ")
    print("="*80)
    
    for idx, row in results_df.iterrows():
        print(f"\nФайл {idx + 1}:")
        print(f"Имя файла: {row['filename']}")
        print(f"Путь: {row['file_path']}")
        print(f"Транскрипция: {row['transcription']}")
        print(f"Статус: {'Успешно' if row['success'] else 'Ошибка'}")
        print("-" * 80)
    
    # Сохраняем результаты
    output_file = "local_transcription_results.csv"
    results_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nРезультаты сохранены в файл: {output_file}")
    
    # Статистика
    successful = results_df['success'].sum()
    total = len(results_df)
    print(f"\nСтатистика:")
    print(f"Успешно обработано: {successful}/{total}")
    print(f"Процент успеха: {successful/total*100:.1f}%")

if __name__ == "__main__":
    main()

