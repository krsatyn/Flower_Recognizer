Поиск похожих растений по изображению с использованием предобученной модели ResNet50 и FastAPI.

---

## 🔧 Установка и запуск

### 1. Клонируйте репозиторий
```bash
git https://github.com/krsatyn/Flower_Recognizer.git
cd Flower_Recognizer
```

### 2. Подготовка эмбеддингов библиотеки
Убедитесь, что у вас есть директория с тестовыми изображениями, например: `flowers/test/`

Далее выполните файл app\create_new_embeding.py:

### 3. Сборка Docker-контейнера
```bash
docker build -t plant-search-app .
```

### 4. Запуск
```bash
docker run -p 8000:8000 plant-search-app
```

---

## 🚀 Использование API

### Эндпоинт: `/search`

**Метод:** `POST`

**Параметры:**
- `file`: изображение растения (jpg)
`

**Пример ответа:**
```json
{
  "app/image_library/daisy/5547758_e12321ea9edfd54_n.jpg": 0.9542,
  "app/image_library/daisy/55423423421311133342344_n.jpg": 0.9374,
  "app/image_library/daisy/554723423423333322dfd54_n.jpg": 0.9121,
  "app/image_library/daisy/55473423443434542234234_n.jpg": 0.8993,
  "app/image_library/daisy/5547758_ee43432a9edfd54_n.jpg": 0.8857
}
```

---

## 📂 Структура проекта
```
plant-search-api/
├── app/
│   └── main.py
│   └── model.py
│   └── utils.py
│   └── create_new_embeding.py
│   └── image_library/
│   │               └── daisy/
│   │               └── dandelion/
│   │               └── rose/
│   │               └── sunflower/
│   │               └── tulip/
│   └── model/
│             └── embeddings/
│             └── weight/
├── requirements.txt
├── Dockerfile
├── .DOCKERIGNORE
├── README.md
├── main.ipynb
├── classification.ipynb
├── data/
```

---

## ✅ TODO / Чек-лист
- [x] Обработка входного изображения
- [x] Построение эмбеддингов
- [x] Топ-5 похожих изображений
- [x] FastAPI и Docker деплой
- [x] Визуализация и пример

---

Готов к запуску и тестированию 🎯