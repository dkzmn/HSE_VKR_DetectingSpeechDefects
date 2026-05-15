## Сервис детекции дефектов речи

Telegram-бот для логопедической диагностики. Пациент отправляет аудиозапись скороговорки, модель оценивает качество произношения по шкале 0–100.

### Архитектура

Два контейнера:
- **bot** — Telegram-бот, SQLite, управление пользователями и доступом (`db.py`, `keyboards.py`, `main.py`)
- **ml_api** — FastAPI-сервис инференса, модель Whisper-small с частичным дообучением

### Структура файлов

```
src/
├── bot/
│   ├── db.py           # БД, константы, вспомогательные функции
│   ├── keyboards.py    # константы кнопок и построители клавиатур
│   ├── main.py         # обработчики и запуск бота
│   ├── requirements.txt
│   └── Dockerfile
├── ml_api/
│   ├── main.py         # FastAPI-эндпоинты и инференс
│   ├── requirements.txt
│   ├── Dockerfile      # CPU-образ (python:3.11-slim)
│   └── Dockerfile.gpu  # GPU-образ (nvidia/cuda:12.1 + CUDA-сборка torch)
├── data/               # SQLite и аудиозаписи (создаётся автоматически)
├── docker-compose.yml
├── docker-compose.gpu.yml  # GPU-override (deploy.resources + Dockerfile.gpu)
└── .env
```

### Быстрый запуск

1. Скопировать `.env.example` в `.env` и заполнить переменные.
2. Убедиться, что папка `../models/whisper_small_finetuned/` содержит `best_ckpt.pt` и `threshold.json`.
3. Скороговорки берутся из `../data/tongue_twisters.csv` автоматически при первом запуске.
4. Если нет, то нужно сделать:
   ```
   dvc pull -r http_yandex
   ```
5. Запустить (CPU):
   ```
   docker compose up --build
   ```

### Запуск с GPU

Требования: NVIDIA GPU, установленные [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) и драйверы.

```
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

GPU-образ (`Dockerfile.gpu`) основан на `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04` и устанавливает CUDA-сборку PyTorch. Переменная `DEVICE` в GPU-режиме по умолчанию `cuda`; можно переопределить в `.env`.

Поведение `ml_api` при разных значениях `DEVICE`:

| `DEVICE`   | Поведение                                            |
|------------|------------------------------------------------------|
| `auto`     | авто: CUDA → MPS → CPU (дефолт, если не задан)      |
| `cuda`     | принудительно GPU (NVIDIA)                           |
| `mps`      | Apple Silicon GPU                                    |
| `cpu`      | принудительно CPU                                    |

`GET /health` при запуске с GPU дополнительно возвращает `gpu_name`, `gpu_memory_allocated_mb`, `gpu_memory_reserved_mb`.

### Переменные окружения (.env)

| Переменная   | Описание                              | Пример                  |
|--------------|---------------------------------------|-------------------------|
| `BOT_TOKEN`  | Токен Telegram-бота                   | `123456:ABC-DEF...`     |
| `ADMIN_IDS`  | Telegram ID администраторов через `,` | `123456789,987654321`   |
| `DEVICE`     | Устройство для инференса              | `auto`, `cpu`, `cuda`, `mps` |

### Роли

| Роль            | Кто назначает          | Возможности                                                              |
|-----------------|------------------------|--------------------------------------------------------------------------|
| **Пациент**     | автоматически при /start | скороговорки, голосовые, свои результаты, управление доступом логопеда |
| **Логопед**     | администратор          | результаты пациентов, выдавших доступ                                   |
| **Администратор** | через `ADMIN_IDS`    | назначение и отзыв роли логопеда                                        |

### Команды бота

**Общие:**
- `/start` — главное меню и список команд
- `/set_name <ФИО>` — указать имя (обязательно при первом входе)
- `/set_letters л р ш` — выбрать контрольные буквы для подбора скороговорок
- `/get_twister` — получить скороговорку
- `/my_results` — последние 5 результатов

**Пациент:**
- `/grant_access <telegram_id>` — выдать доступ логопеду
- `/revoke_access <telegram_id>` — отозвать доступ

**Логопед / Администратор:**
- `/patients` — список пациентов с выданным доступом
- `/patient_results <telegram_id>` — результаты конкретного пациента

**Администратор:**
- `/set_role <telegram_id> <admin|therapist|patient>` — изменить роль пользователя

### ML API

`POST /score` — оценка произношения.

Параметры: 
- `audio` (WAV), 
- `twister_id`, 
- `letters` (JSON-массив флагов букв), 
- `duration`, 
- `n_speakers`.

Ответ: 
- `score` (0–100), 
- `label` (`good`/`bad`), 
- `proba`, 
- `threshold`, 
- `model_version`.

`GET /health` — проверка работоспособности. Возвращает `device` (тип устройства), а при CUDA — `gpu_name`, `gpu_memory_allocated_mb`, `gpu_memory_reserved_mb`.

### Примеры работы бота

Отправка голосовых сообщений и получение результата
![image](../images/example_1.jpeg)

Выбор контрольных букв
![image](../images/example_2.jpeg)

Просмотр результатов
![image](../images/example_3.jpg)