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
from langgraph.prebuilt import create_react_agent
from langchain_gigachat import GigaChat
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langgraph.prebuilt import ToolNode, tools_condition

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
    tools: List[Dict[str, Any]]  # Результаты от ToolNode
    needs_revision: bool  # Флаг необходимости доработки контента
    revision_count: int  # Счетчик итераций доработки

# Инициализируем GigaChat с Langfuse callback и увеличенными таймаутами
llm = GigaChat(
    credentials=os.getenv("GIGACHAT_API_KEY"),
    verify_ssl_certs=False,
    timeout=120.0,  # Увеличиваем таймаут до 2 минут
    request_timeout=120.0,  # Таймаут для HTTP-запросов
    max_retries=3,  # Количество попыток при ошибке
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
    """Агент-писатель: создает контент на основе анализа и учитывает критику"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Ты талантливый писатель. Создай качественный, структурированный контент на основе предоставленного анализа.
        
        Если есть критика от редактора, обязательно учти все замечания и улучши контент.
        Пиши ясно, логично и увлекательно."""),
        ("user", """Тема: {topic}
Анализ: {analysis}
Предыдущий контент: {content}
Критика редактора: {feedback}

Если это первая версия (нет предыдущего контента), создай новый материал. 
Если есть критика, улучши существующий контент, учитывая все замечания редактора.""")
    ])
    
    def writer(state: AgentState) -> AgentState:
        print("DEBUG: Писатель начал работу")
        
        # Проверяем, есть ли уже контент (это доработка)
        is_revision = bool(state.get("content", "").strip())
        if is_revision:
            # Увеличиваем счетчик доработок
            state["revision_count"] = state.get("revision_count", 0) + 1
            print(f"DEBUG: Писатель дорабатывает контент на основе критики (итерация {state['revision_count']})")
        else:
            # Инициализируем счетчик для первого контента
            state["revision_count"] = 0
            print("DEBUG: Писатель создает первичный контент")
            
        messages = prompt.format_messages(
            topic=state["topic"],
            analysis=state["analysis"],
            content=state.get("content", ""),
            feedback=state.get("feedback", "")
        )
        try:
            response = llm.invoke(messages)
            state["content"] = response.content
            state["messages"].append(AIMessage(content=f"Контент: {response.content}"))
            print(f"DEBUG: Писатель получил ответ длиной {len(response.content)} символов")
            print("DEBUG: Писатель завершил работу")
        except Exception as e:
            print(f"DEBUG: Ошибка писателя: {e}")
            state["content"] = f"Ошибка создания контента: {e}"
        return state
    
    return writer

def create_critic_agent():
    """Агент-критик: оценивает контент и принимает решение о необходимости доработки"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Ты строгий критик и редактор. Твоя задача - проанализировать контент и принять решение о его качестве.

Критерии оценки:
- Структура и логика изложения
- Полнота раскрытия темы  
- Качество и ясность текста
- Соответствие теме

ВАЖНО: В конце своей критики ты ДОЛЖЕН явно указать одно из решений:
- "РЕШЕНИЕ: ДОРАБОТАТЬ" - если контент требует значительных улучшений
- "РЕШЕНИЕ: ПРИНЯТЬ" - если контент достаточно хорош и готов к финализации

Дай конструктивную обратную связь с конкретными предложениями по улучшению."""),
        ("user", "Тема: {topic}\nАнализ: {analysis}\nКонтент: {content}\n\nОцени этот контент и дай подробную критику с решением о дальнейших действиях.")
    ])
    
    def critic(state: AgentState) -> AgentState:
        print("DEBUG: Критик начал работу")
        messages = prompt.format_messages(
            topic=state["topic"],
            analysis=state["analysis"],
            content=state["content"]
        )
        response = llm.invoke(messages)
        state["feedback"] = response.content
        state["messages"].append(AIMessage(content=f"Критика: {response.content}"))
        
        # Определяем решение критика
        if "РЕШЕНИЕ: ДОРАБОТАТЬ" in response.content:
            state["needs_revision"] = True
            print("DEBUG: Критик решил, что контент нужно доработать")
        elif "РЕШЕНИЕ: ПРИНЯТЬ" in response.content:
            state["needs_revision"] = False
            print("DEBUG: Критик принял контент")
        else:
            # По умолчанию считаем, что нужна доработка, если решение неясно
            state["needs_revision"] = True
            print("DEBUG: Критик не дал четкого решения, отправляем на доработку")
            
        print("DEBUG: Критик завершил работу")
        return state
    
    return critic



def create_tools_agent():
    """Агент-инструментарий с использованием ToolNode"""
    
    @tool
    def analyze_text(text: str) -> str:
        """
        Анализирует текст и возвращает полезную информацию о нем.
        
        Args:
            text: Текст для анализа
        """
        try:
            # Подсчет слов
            words = text.split()
            word_count = len(words)
            
            # Простой анализ настроения
            positive_words = ['хорошо', 'отлично', 'прекрасно', 'замечательно', 'удивительно', 'великолепно']
            negative_words = ['плохо', 'ужасно', 'отвратительно', 'ужас', 'кошмар', 'отвратительно']
            
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                sentiment = "Положительная"
            elif negative_count > positive_count:
                sentiment = "Отрицательная"
            else:
                sentiment = "Нейтральная"
            
            # Подсчет предложений (простой способ)
            sentences = text.split('.')
            sentence_count = len([s for s in sentences if s.strip()])
            
            # Средняя длина предложения
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            
            # Текущее время
            from datetime import datetime
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Формируем результат анализа
            analysis_result = f"""
📊 АНАЛИЗ ТЕКСТА:
⏰ Время анализа: {current_time}
📝 Количество слов: {word_count}
📄 Количество предложений: {sentence_count}
📏 Средняя длина предложения: {avg_sentence_length:.1f} слов
😊 Эмоциональная окраска: {sentiment}
            """.strip()
            
            return analysis_result
            
        except Exception as e:
            return f"Ошибка при анализе текста: {str(e)}"
    
    # Создаем инструменты
    tools = [analyze_text]
    
    def tools_agent(state: AgentState) -> AgentState:
        print("DEBUG: ToolNode агент-инструментарий начал работу")
        
        try:
            # Всегда анализируем контент с помощью инструмента
            text_to_analyze = state["content"]
            print(f"DEBUG: Анализируем текст длиной {len(text_to_analyze)} символов")
            
            # Вызываем инструмент напрямую
            analysis_result = analyze_text.invoke({"text": text_to_analyze})
            
            # Обогащаем контент результатами анализа
            enriched_content = f"""
{state["content"]}

---
{analysis_result}
            """.strip()
            
            # Обновляем состояние
            state["tools_used"] = ["analyze_text"]
            state["tool_results"] = {"text_analysis": analysis_result}
            state["content"] = enriched_content
            state["messages"].append(AIMessage(content=f"Обогащенный контент: {enriched_content[:200]}..."))
            
            print(f"DEBUG: ToolNode агент использовал инструмент: analyze_text")
            print(f"DEBUG: Результат анализа получен")
            
        except Exception as e:
            print(f"DEBUG: Ошибка ToolNode агента: {e}")
            # Fallback к простому анализу
            from datetime import datetime
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            word_count = len(state["content"].split())
            
            tools_results = {
                "analyze_text": f"Простой анализ: {word_count} слов, Время: {current_time}"
            }
            tools_used = ["analyze_text"]
            
            state["tools_used"] = tools_used
            state["tool_results"] = tools_results
            
            enriched_content = f"""
{state["content"]}

---
📊 ПРОСТОЙ АНАЛИЗ:
⏰ {current_time}
📝 Количество слов: {word_count}
❌ Произошла ошибка при использовании ToolNode агента
            """.strip()
            
            state["content"] = enriched_content
        
        print("DEBUG: ToolNode агент-инструментарий завершил работу")
        return state
    
    return tools_agent

# Функция принятия решения для условного перехода
def should_continue(state: AgentState) -> str:
    """Определяет, нужна ли доработка контента или можно финализировать"""
    max_revisions = 3  # Максимальное количество доработок
    current_revisions = state.get("revision_count", 0)
    
    if state.get("needs_revision", True) and current_revisions < max_revisions:
        print(f"DEBUG: Отправляем контент на доработку писателю (итерация {current_revisions + 1}/{max_revisions})")
        return "writer"
    else:
        if current_revisions >= max_revisions:
            print(f"DEBUG: Достигнут лимит доработок ({max_revisions}), переходим к инструментам")
        else:
            print("DEBUG: Контент принят, переходим к инструментам")
        return "tools"

# Создаем граф агентов
def create_agent_graph():
    """Создает граф мультиагентной системы с простой нодой для инструментов"""
    
    # Создаем граф
    workflow = StateGraph(AgentState)
    
    # Создаем простую функцию для анализа текста
    def analyze_text_node(state: AgentState) -> AgentState:
        """Нода для анализа текста с помощью инструмента"""
        print("DEBUG: Анализ текста с помощью инструмента")
        
        try:
            # Создаем инструмент
            @tool
            def analyze_text(text: str) -> str:
                """Анализирует текст и возвращает статистику"""
                try:
                    words = text.split()
                    word_count = len(words)
                    sentences = text.split('.')
                    sentence_count = len([s for s in sentences if s.strip()])
                    
                    from datetime import datetime
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    return f"""
📊 АНАЛИЗ ТЕКСТА:
⏰ Время анализа: {current_time}
📝 Количество слов: {word_count}
📄 Количество предложений: {sentence_count}
                    """.strip()
                except Exception as e:
                    return f"Ошибка при анализе текста: {str(e)}"
            
            # Анализируем контент
            text_to_analyze = state["content"]
            analysis_result = analyze_text.invoke({"text": text_to_analyze})
            
            # Обогащаем контент результатами анализа
            enriched_content = f"""
{state["content"]}

---
{analysis_result}
            """.strip()
            
            # Обновляем состояние
            state["tools_used"] = ["analyze_text"]
            state["tool_results"] = {"analyze_text": analysis_result}
            state["content"] = enriched_content
            state["messages"].append(AIMessage(content=f"Обогащенный контент: {enriched_content[:200]}..."))
            
            print(f"DEBUG: Контент обогащен результатами анализа")
            
        except Exception as e:
            print(f"DEBUG: Ошибка анализа текста: {e}")
            # Fallback к простому анализу
            from datetime import datetime
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            word_count = len(state["content"].split())
            
            tools_results = {
                "text_analysis": f"Простой анализ: {word_count} слов, Время: {current_time}"
            }
            tools_used = ["analyze_text"]
            
            state["tools_used"] = tools_used
            state["tool_results"] = tools_results
            
            enriched_content = f"""
{state["content"]}

---
📊 ПРОСТОЙ АНАЛИЗ:
⏰ {current_time}
📝 Количество слов: {word_count}
❌ Произошла ошибка при использовании инструмента анализа
            """.strip()
            
            state["content"] = enriched_content
        
        return state
    
    # Добавляем узлы (агентов)
    workflow.add_node("analyst", create_analyst_agent())
    workflow.add_node("writer", create_writer_agent())
    workflow.add_node("critic", create_critic_agent())
    workflow.add_node("tools", analyze_text_node)
    
    # Определяем поток выполнения
    workflow.set_entry_point("analyst")
    workflow.add_edge("analyst", "writer")
    workflow.add_edge("writer", "critic")
    
    # Условный переход от критика
    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "writer": "writer",  # Если нужна доработка - обратно к писателю
            "tools": "tools"     # Если контент принят - к инструментам
        }
    )
    
    workflow.add_edge("tools", END)
    
    return workflow.compile().with_config(callbacks=[langfuse_handler])

# Функция для запуска мультиагентной системы
def run_multi_agent_system(topic: str) -> Dict[str, Any]:
    """
    Запускает мультиагентную систему для обработки заданной темы
    
    Args:
        topic: Тема для обработки
        use_langfuse: Использовать ли Langfuse трассировку
        
    Returns:
        Результат работы системы
    """
    # Создаем граф
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
        tool_results={},
        tools=[],  # Результаты от ToolNode
        needs_revision=False,
        revision_count=0
    )
    
    # Запускаем систему
    result = graph.invoke(initial_state)
    
    return {
        "topic": result["topic"],
        "analysis": result["analysis"],
        "content": result["content"],
        "feedback": result["feedback"],
        "final_result": result["content"],  # Используем content как финальный результат
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
        result = run_multi_agent_system(test_topic)
        
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
        print(result["content"])
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
