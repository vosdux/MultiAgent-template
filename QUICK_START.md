# 🚀 Быстрый старт - Мультиагентная система

## ⚡ За 5 минут к работающей системе

### 1. Подготовка окружения
```bash
# Клонируйте репозиторий
git clone <repository-url>
cd AI-Agent

# Создайте виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# Установите зависимости
pip install -r requirements.txt
```

### 2. Настройка Langfuse (опционально)
```bash
# Быстрое развертывание мониторинга
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up

# Откройте http://localhost:3000 и создайте проект
```

### 3. Настройка API ключей
Создайте файл `config.env`:
```env
GIGACHAT_API_KEY=ваш_ключ_gigachat

# Langfuse (опционально - для мониторинга)
LANGFUSE_PUBLIC_KEY=ваш_публичный_ключ_langfuse
LANGFUSE_SECRET_KEY=ваш_секретный_ключ_langfuse
LANGFUSE_HOST=http://localhost:3000
```

### 4. Запуск системы
```bash
python agents.py
```

### 5. Выберите режим работы
- **1** - Автоматический (рекомендуется для начала)
- **2** - Интерактивный (с возможностью вмешательства)
- **3** - Восстановление сессии

## 🎯 Что вы получите

После запуска система создаст:
- 📊 Детальный анализ темы
- ✍️ Качественный контент
- 🔍 Конструктивную критику
- 🎯 Финальную версию
- 📈 Дополнительную аналитику

## 🔧 Первая настройка

### Для VS Code/Cursor
Создайте `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.terminal.activateEnvironment": true
}
```

### Проверка установки
```bash
# Проверьте, что все работает
python -c "import langchain, langgraph, langfuse; print('✅ Все библиотеки установлены')"
```

## 🚨 Частые проблемы

### "ModuleNotFoundError: No module named 'dotenv'"
```bash
# Решение: активируйте виртуальное окружение
source venv/bin/activate
```

### "Unauthorized" от GigaChat
```bash
# Решение: проверьте API ключ в config.env
cat config.env
```

### Редактор не видит библиотеки
```bash
# Решение: выберите правильный Python интерпретатор
# Cmd+Shift+P → "Python: Select Interpreter" → ./venv/bin/python
```

## 🎮 Попробуйте разные режимы

### Автоматический режим
```bash
python agents.py
# Выберите: 1
# Введите тему: "Искусственный интеллект в образовании"
```

### Интерактивный режим
```bash
python agents.py
# Выберите: 2
# На каждом этапе можете:
# - y: продолжить
# - n: остановиться
# - edit: отредактировать результат
```

## 📊 Мониторинг

### Langfuse дашборд (если настроен)
```bash
# 1. Убедитесь, что Langfuse запущен
cd langfuse
docker compose up

# 2. Откройте дашборд
open http://localhost:3000
```

Что вы увидите:
- **Трассировки** выполнения агентов
- **Производительность** каждого шага
- **Использование токенов** LLM
- **Ошибки** и проблемы

### Локальная отладка
```bash
# Включите отладочную информацию
export DEBUG=1
python agents.py
```

## 🔄 Следующие шаги

1. **Изучите код** - посмотрите на `agents.py`
2. **Добавьте свои агенты** - создайте новых специалистов
3. **Настройте промпты** - адаптируйте под свои задачи
4. **Интегрируйте с API** - подключите внешние сервисы

## 💡 Советы

- Начните с автоматического режима
- Сохраняйте Thread ID для восстановления
- Используйте интерактивный режим для сложных задач
- Мониторьте производительность через Langfuse

## 🆘 Нужна помощь?

- 📖 Полная документация: `README.md`
- 📊 Настройка мониторинга: `LANGFUSE_SETUP.md`
- 💾 Работа с состоянием: `MEMORY_SAVER_GUIDE.md`
- 🐛 Проблемы: создайте Issue в GitHub

---

**Готово!** Ваша мультиагентная система работает! 🎉
