# KAG POC: архитектура извлечения

## Цель
Построить Knowledge‑Augmented Graph из мульти‑источникового контента (текст, код, SQL) и
поддержать улучшение context mining для LLM.

## Сущности графа (property graph)
- File, Section
- Term, Entity
- Class, Function, Method, Module, Symbol
- Table, Column

## Типы связей
- defined_in, mentions, defines
- is_a, part_of, same_as
- imports, calls, defines_method, instantiates
- has_column, fk

## Пайплайн (offline)
1) Ingestion
   - Markdown, Python, Java, SQL DDL
2) Normalization
   - токенизация/нормализация терминов
3) Extraction (уровни сигнала)
   - L0: правила (definitions, part_of, is_a)
   - L0: Table Extraction (CSV -> Graph schema/data)
   - L1: NER (spaCy) для Entity
   - L1.5: SOTA NER (GLiNER) + Relation Extraction (REBEL)
   - L2: LLM relation extraction
4) Graph Build
   - дедупликация узлов по (type,label)
   - дедупликация ребер по (src,type,tgt)
5) Output
   - JSON nodes/edges/graph + report

## Как выделяются термины и связи

### 1) Правила (L0)
Извлекаем термины из Markdown:
- заголовки `#`
- **bold**
- `inline code`

Связи из текстовых паттернов:
- `X is a Y` → is_a
- `X — это Y` → is_a
- `X is_a Y` → is_a
- `X part_of Y` → part_of
- `X corresponds to Y` / `X соответствует Y` → same_as

Извлечение из кода:
- Python AST: классы, функции, импортируемые модули, вызовы
- Java regex: классы, методы, new‑инстансы

Из SQL:
- таблицы и колонки
- foreign key → fk

### 2) NER (L1)
Используется spaCy (многоязычный режим):
- модель выбирается из списка `--spacy-models`
- найденные сущности → узлы Entity
- связь `mentions` от источника (File/Class/Function) к Entity

### 3) SOTA Extraction (L1.5)
Используются opensource модели:
- **GLiNER** (`--gliner`): Zero-shot NER для выделения сущностей (Person, Org, Loc, etc.).
- **REBEL** (`--rebel`): End-to-end Relation Extraction для генерации троек (head, relation, tail).

### 4) Table Extraction
Обработка CSV файлов:
- Таблица → узел `Table`
- Колонки → узлы `Column`
- Строки → узлы `Entity` (используя PK heuristic)
- Связи `has_column`, `contained_in`, атрибуты.

### 5) LLM relation extraction (L2)
Опционально, включается флагом `--llm`.
- запрос к OpenAI‑compatible endpoint через `scripts/llm_client.py`
- LLM возвращает JSON со списком троек
- связь добавляется как `head -relation-> tail` с `source=llm`

## Мульти‑язычность
- Правила покрывают EN/RU (is_a, соответствует)
- NER модель выбирается мульти‑язычная (xx) или RU/EN
- LLM слой не зависит от языка

## Границы и улучшения
- Heuristic‑L0 даёт высокий recall, но шумный
- NER требует обученной модели под домен
- LLM лучше выявляет сложные отношения, но стоит дороже
- Для prod: entity linking, relation schema, confidence scoring, graph DB

## Конфигурация
- NER: `--ner --spacy-models "en_core_web_sm,ru_core_news_sm,xx_ent_wiki_sm"`
- LLM: `--llm --llm-max-calls 5`
- LLM env:
  - `KAG_LLM_ENDPOINT`
  - `KAG_LLM_API_KEY`
  - `KAG_LLM_MODEL`
