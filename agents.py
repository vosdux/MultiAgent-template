import os
import json
import requests
from datetime import datetime
from typing import Dict, List, Any, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_gigachat import GigaChat
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

# Загружаем переменные окружения
load_dotenv("config.env")

# Инициализируем Langfuse для мониторинга
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3000")
)

# Создаем callback handler для LangChain
langfuse_handler = CallbackHandler(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    update_trace=True
)

# Определяем простые функции инструментов
def get_current_time_simple() -> str:
    """Получает текущее время и дату"""
    return f"Текущее время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

def calculate_word_count_simple(text: str) -> str:
    """Подсчитывает количество слов в тексте"""
    words = text.split()
    return f"Количество слов в тексте: {len(words)}"

def get_weather_info_simple(city: str = "Moscow") -> str:
    """Получает информацию о погоде в указанном городе (мок-данные)"""
    weather_data = {
        "Moscow": {"temp": "15°C", "condition": "Облачно", "humidity": "65%"},
        "Saint Petersburg": {"temp": "12°C", "condition": "Дождь", "humidity": "80%"},
        "Novosibirsk": {"temp": "8°C", "condition": "Солнечно", "humidity": "45%"},
        "Yekaterinburg": {"temp": "10°C", "condition": "Переменная облачность", "humidity": "60%"}
    }
    
    if city in weather_data:
        data = weather_data[city]
        return f"Погода в {city}: {data['temp']}, {data['condition']}, влажность: {data['humidity']}"
    else:
        return f"Погода в {city}: 20°C, Солнечно, влажность: 50% (мок-данные)"

def translate_text_simple(text: str, target_language: str = "en") -> str:
    """Переводит текст на указанный язык (мок-перевод)"""
    translations = {
        "en": f"[EN] {text[:100]}...",
        "es": f"[ES] {text[:100]}...",
        "fr": f"[FR] {text[:100]}...",
        "de": f"[DE] {text[:100]}..."
    }
    return translations.get(target_language, f"[{target_language.upper()}] {text[:100]}...")

def analyze_sentiment_simple(text: str) -> str:
    """Анализирует эмоциональную окраску текста"""
    positive_words = ["хорошо", "отлично", "прекрасно", "великолепно", "замечательно", "позитивно", "успешно"]
    negative_words = ["плохо", "ужасно", "отвратительно", "негативно", "грустно", "проблема", "ошибка"]
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return "Эмоциональная окраска: Позитивная"
    elif negative_count > positive_count:
        return "Эмоциональная окраска: Негативная"
    else:
        return "Эмоциональная окраска: Нейтральная"

def generate_summary_simple(text: str, max_length: int = 100) -> str:
    """Генерирует краткое резюме текста"""
    words = text.split()
    if len(words) <= max_length:
        return text
    else:
        summary = " ".join(words[:max_length]) + "..."
        return f"Краткое резюме: {summary}"

# Определяем инструменты LangChain (для совместимости)
@tool
def get_current_time() -> str:
    """Получает текущее время и дату"""
    return get_current_time_simple()

@tool
def calculate_word_count(text: str) -> str:
    """Подсчитывает количество слов в тексте"""
    return calculate_word_count_simple(text)

@tool
def get_weather_info(city: str = "Moscow") -> str:
    """Получает информацию о погоде в указанном городе (мок-данные)"""
    return get_weather_info_simple(city)

@tool
def translate_text(text: str, target_language: str = "en") -> str:
    """Переводит текст на указанный язык (мок-перевод)"""
    return translate_text_simple(text, target_language)

@tool
def analyze_sentiment(text: str) -> str:
    """Анализирует эмоциональную окраску текста"""
    return analyze_sentiment_simple(text)

@tool
def generate_summary(text: str, max_length: int = 100) -> str:
    """Генерирует краткое резюме текста"""
    return generate_summary_simple(text, max_length)

# Список всех доступных инструментов
available_tools = [
    get_current_time,
    calculate_word_count,
    get_weather_info,
    translate_text,
    analyze_sentiment,
    generate_summary
]

# Определяем структуру состояния
class AgentState(TypedDict):
    messages: List[Any]
    topic: str
    analysis: str
    content: str
    feedback: str
    final_result: str
    tools_used: List[str]  # Список использованных инструментов
    tool_results: Dict[str, str]  # Результаты работы инструментов

# Инициализируем GigaChat с Langfuse callback
llm = GigaChat(
    credentials=os.getenv("GIGACHAT_API_KEY"),
    verify_ssl_certs=False,
)

# Определяем агентов
def create_analyst_agent():
    """Агент-аналитик: анализирует тему и создает план"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Ты опытный аналитик. Твоя задача - проанализировать заданную тему и создать детальный план работы.
        Будь конкретным и структурированным в своем анализе."""),
        ("user", "Проанализируй тему: {topic}")
    ])
    
    def analyst(state: AgentState) -> AgentState:
        print("DEBUG: Аналитик начал работу")
        messages = prompt.format_messages(topic=state["topic"])
        print(f"DEBUG: Аналитик отправил запрос: {messages[-1].content[:100]}...")
        try:
            response = llm.invoke(messages)
            state["analysis"] = response.content
            state["messages"].append(AIMessage(content=f"Анализ: {response.content}"))
            print(f"DEBUG: Аналитик получил ответ длиной {len(response.content)} символов")
        except Exception as e:
            print(f"DEBUG: Ошибка аналитика: {e}")
            state["analysis"] = f"Ошибка анализа: {e}"
        print("DEBUG: Аналитик завершил работу")
        return state
    
    return analyst

def create_writer_agent():
    """Агент-писатель: создает контент на основе анализа"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Ты талантливый писатель. Создай качественный контент на основе предоставленного анализа.
        Будь креативным, но следуй структуре и рекомендациям аналитика."""),
        ("user", "Анализ: {analysis}\n\nСоздай контент на основе этого анализа.")
    ])
    
    def writer(state: AgentState) -> AgentState:
        messages = prompt.format_messages(analysis=state["analysis"])
        try:
            response = llm.invoke(messages)
            state["content"] = response.content
            state["messages"].append(AIMessage(content=f"Контент: {response.content}"))
            print(f"DEBUG: Писатель получил ответ длиной {len(response.content)} символов")
        except Exception as e:
            print(f"DEBUG: Ошибка писателя: {e}")
            state["content"] = f"Ошибка создания контента: {e}"
        return state
    
    return writer

def create_critic_agent():
    """Агент-критик: оценивает и улучшает контент"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Ты строгий критик и редактор. Оцени контент и предложи улучшения.
        Будь конструктивным и конкретным в своих замечаниях."""),
        ("user", "Тема: {topic}\nАнализ: {analysis}\nКонтент: {content}\n\nОцени этот контент и предложи улучшения.")
    ])
    
    def critic(state: AgentState) -> AgentState:
        messages = prompt.format_messages(
            topic=state["topic"],
            analysis=state["analysis"],
            content=state["content"]
        )
        response = llm.invoke(messages)
        state["feedback"] = response.content
        state["messages"].append(AIMessage(content=f"Критика: {response.content}"))
        return state
    
    return critic

def create_finalizer_agent():
    """Финальный агент: создает итоговый результат"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Ты финальный редактор. Создай итоговую версию контента, учитывая все замечания критика.
        Объедини лучшие части и создай финальный результат."""),
        ("user", "Тема: {topic}\nАнализ: {analysis}\nКонтент: {content}\nКритика: {feedback}\n\nСоздай финальную версию.")
    ])
    
    def finalizer(state: AgentState) -> AgentState:
        messages = prompt.format_messages(
            topic=state["topic"],
            analysis=state["analysis"],
            content=state["content"],
            feedback=state["feedback"]
        )
        response = llm.invoke(messages)
        state["final_result"] = response.content
        state["messages"].append(AIMessage(content=f"Финальный результат: {response.content}"))
        return state
    
    return finalizer

def create_tools_agent():
    """Агент-инструментарий: использует различные инструменты для обогащения контента"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Ты агент-инструментарий. Твоя задача - использовать доступные инструменты для обогащения и улучшения контента.
        
        Доступные инструменты:
        - get_current_time: получить текущее время
        - calculate_word_count: подсчитать количество слов в тексте
        - get_weather_info: получить информацию о погоде
        - translate_text: перевести текст на другой язык
        - analyze_sentiment: проанализировать эмоциональную окраску текста
        - generate_summary: создать краткое резюме текста
        
        Используй инструменты для добавления полезной информации к контенту."""),
        ("user", "Тема: {topic}\nАнализ: {analysis}\nКонтент: {content}\nКритика: {feedback}\n\nИспользуй инструменты для обогащения контента.")
    ])
    
    def tools_agent(state: AgentState) -> AgentState:
        print("DEBUG: Агент-инструментарий начал работу")
        
        # Используем несколько инструментов
        tools_results = {}
        tools_used = []
        
        # Получаем текущее время
        current_time = get_current_time_simple()
        tools_results["current_time"] = current_time
        tools_used.append("get_current_time")
        
        # Подсчитываем количество слов в контенте
        word_count = calculate_word_count_simple(state["content"])
        tools_results["word_count"] = word_count
        tools_used.append("calculate_word_count")
        
        # Анализируем эмоциональную окраску
        sentiment = analyze_sentiment_simple(state["content"])
        tools_results["sentiment"] = sentiment
        tools_used.append("analyze_sentiment")
        
        # Создаем краткое резюме
        summary = generate_summary_simple(state["content"], 50)
        tools_results["summary"] = summary
        tools_used.append("generate_summary")
        
        # Получаем информацию о погоде
        weather = get_weather_info_simple("Moscow")
        tools_results["weather"] = weather
        tools_used.append("get_weather_info")
        
        # Обновляем состояние
        state["tools_used"] = tools_used
        state["tool_results"] = tools_results
        
        # Создаем обогащенный контент
        enriched_content = f"""
{state["content"]}

---
📊 ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:
⏰ {current_time}
📝 {word_count}
😊 {sentiment}
🌤️ {weather}
📋 {summary}
        """.strip()
        
        state["content"] = enriched_content
        state["messages"].append(AIMessage(content=f"Обогащенный контент с инструментами: {enriched_content[:200]}..."))
        
        print("DEBUG: Агент-инструментарий завершил работу")
        return state
    
    return tools_agent

# Создаем граф агентов
def create_agent_graph():
    """Создает граф мультиагентной системы"""
    
    # Создаем граф
    workflow = StateGraph(AgentState)
    
    # Добавляем узлы (агентов)
    workflow.add_node("analyst", create_analyst_agent())
    workflow.add_node("writer", create_writer_agent())
    workflow.add_node("critic", create_critic_agent())
    workflow.add_node("tools", create_tools_agent())
    workflow.add_node("finalizer", create_finalizer_agent())
    
    # Определяем поток выполнения
    workflow.set_entry_point("analyst")
    workflow.add_edge("analyst", "writer")
    workflow.add_edge("writer", "critic")
    workflow.add_edge("critic", "tools")
    workflow.add_edge("tools", "finalizer")
    workflow.add_edge("finalizer", END)
    
    return workflow.compile().with_config(callbacks=[langfuse_handler])

# Создаем граф агентов с Langfuse трассировкой
def create_agent_graph_with_langfuse():
    """Создает граф мультиагентной системы с Langfuse трассировкой"""
    
    # Создаем граф
    workflow = StateGraph(AgentState)
    
    # Добавляем узлы (агентов)
    workflow.add_node("analyst", create_analyst_agent())
    workflow.add_node("writer", create_writer_agent())
    workflow.add_node("critic", create_critic_agent())
    workflow.add_node("finalizer", create_finalizer_agent())
    
    # Определяем поток выполнения
    workflow.set_entry_point("analyst")
    workflow.add_edge("analyst", "writer")
    workflow.add_edge("writer", "critic")
    workflow.add_edge("critic", "finalizer")
    workflow.add_edge("finalizer", END)
    
    # Компилируем с Langfuse callback
    return workflow.compile(checkpointer=None, interrupt_before=["analyst", "writer", "critic", "finalizer"])

# Функция для запуска мультиагентной системы
def run_multi_agent_system(topic: str, use_langfuse: bool = True) -> Dict[str, Any]:
    """
    Запускает мультиагентную систему для обработки заданной темы
    
    Args:
        topic: Тема для обработки
        use_langfuse: Использовать ли Langfuse трассировку
        
    Returns:
        Результат работы системы
    """
    # Создаем граф
    if use_langfuse:
        graph = create_agent_graph_with_langfuse()
    else:
        graph = create_agent_graph()
    
    # Инициализируем состояние
    initial_state = AgentState(
        messages=[],
        topic=topic,
        analysis="",
        content="",
        feedback="",
        final_result="",
        tools_used=[],
        tool_results={}
    )
    
    # Запускаем систему
    if use_langfuse:
        print("DEBUG: Запуск с Langfuse мониторингом...")
        # Langfuse автоматически трассирует LLM вызовы через callback handler
        result = graph.invoke(initial_state)
        print("DEBUG: Граф выполнен успешно с Langfuse")
    else:
        print("DEBUG: Запуск без Langfuse...")
        result = graph.invoke(initial_state)
    
    return {
        "topic": result["topic"],
        "analysis": result["analysis"],
        "content": result["content"],
        "feedback": result["feedback"],
        "final_result": result["final_result"],
        "messages": result["messages"],
        "tools_used": result.get("tools_used", []),
        "tool_results": result.get("tool_results", {})
    }

# Пример использования
if __name__ == "__main__":
    # Пример темы для обработки
    test_topic = "Искусственный интеллект в современном образовании"
    
    print("🚀 Запуск мультиагентной системы...")
    print(f"📝 Тема: {test_topic}")
    print("-" * 50)
    
    # Проверяем настройки Langfuse
    langfuse_enabled = all([
        os.getenv("LANGFUSE_PUBLIC_KEY"),
        os.getenv("LANGFUSE_SECRET_KEY"),
        os.getenv("LANGFUSE_HOST")
    ])
    
    if langfuse_enabled:
        print("📊 Langfuse трассировка включена")
        print(f"🌐 Langfuse Host: {os.getenv('LANGFUSE_HOST')}")
    else:
        print("⚠️ Langfuse трассировка отключена (отсутствуют настройки)")
    
    print("-" * 50)
    
    try:
        # Используем Langfuse только для мониторинга, не для трассировки графа
        result = run_multi_agent_system(test_topic, use_langfuse=False)
        
        print("\n📊 АНАЛИЗ:")
        print(result["analysis"])
        print("\n" + "="*50)
        
        print("\n✍️ КОНТЕНТ:")
        print(result["content"])
        print("\n" + "="*50)
        
        print("\n🔍 КРИТИКА:")
        print(result["feedback"])
        print("\n" + "="*50)
        
        print("\n🎯 ФИНАЛЬНЫЙ РЕЗУЛЬТАТ:")
        print(result["final_result"])
        print("\n" + "="*50)
        
        # Показываем информацию об использованных инструментах
        if result.get("tools_used"):
            print("\n🔧 ИСПОЛЬЗОВАННЫЕ ИНСТРУМЕНТЫ:")
            for tool_name in result["tools_used"]:
                print(f"   - {tool_name}")
            
            print("\n📊 РЕЗУЛЬТАТЫ РАБОТЫ ИНСТРУМЕНТОВ:")
            for tool_name, tool_result in result.get("tool_results", {}).items():
                print(f"   {tool_name}: {tool_result}")
        
        print("\n" + "="*50)
        print("\n✅ Мультиагентная система завершила работу!")
        
        if langfuse_enabled:
            print("\n📈 Трассировка сохранена в Langfuse")
            print("🔗 Проверьте дашборд для детального анализа")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        if langfuse_enabled:
            print("💡 Проверьте настройки Langfuse в config.env")
