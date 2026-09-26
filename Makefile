.PHONY: all download prepare-data run-embedding run-string-equiv run-llm run-llm-deepseek \
        run-llm-local compare compare-full consistency test ablation retrieval stats gnn \
        gnn-ontology gnn-baseline repeats provenance docx lint viewer viz clean

# This hack prevents creation of ugly `__pycache__` directories
# in the source tree.
export PYTHONPYCACHEPREFIX:=/tmp/pythoncache

all: prepare-data run-embedding run-string-equiv

# ── Загрузка и подготовка данных ────────────────────────────────────────

# Загружает 7 онтологий OAEI 2025 Conference track и 21 эталонный alignment.
# Требуется только один раз; результаты сохраняются в `data/raw/oaei/`.
download:
	@echo "Downloading OAEI 2025 Conference track data..."
	mkdir -p data/raw/oaei
	for onto in cmt.owl confof.owl Conference.owl edas.owl ekaw.owl iasted.owl sigkdd.owl; do \
		name=$$(basename "$$onto" .owl); \
		wget -q -O "data/raw/oaei/$$name.owl" \
			"http://oaei.ontologymatching.org/2025/conference/data/$$onto" || exit 1; \
	done
	wget -q -O data/raw/oaei/reference-alignment.zip \
		"http://oaei.ontologymatching.org/2025/conference/data/reference-alignment.zip"
	cd data/raw/oaei && unzip -o reference-alignment.zip && rm reference-alignment.zip
	@echo "Download complete."

# Парсит OWL-файлы в JSON-таксономии и генерирует синтетические вариации.
# Читает data/raw/oaei/, пишет data/processed/oaei/ и data/processed/synthetic/.
prepare-data:
	uv run -m src.prepare_data

# ── Эксперименты (быстрые) ───────────────────────────────────────────────

# Эмбеддинговый подход: LaBSE + косинусная близость.  ~1 минута.
run-embedding:
	ALL_PROXY= uv run -m src.runners.embedding

# StringEquiv baseline: регистронезависимое сравнение меток.  Мгновенно.
run-string-equiv:
	uv run -m src.runners.string_equiv

# ── Эксперименты (LLM — требуют API-ключей) ─────────────────────────────

# LLM/gpt-4.1-mini (kodikrouter).  ~3-5 минут на шесть пар (дефолт раннера).
# Требует KODIKROUTER_API_KEY.
run-llm:
	ALL_PROXY= uv run -m src.runners.llm

# LLM/deepseek-v4-flash (DeepSeek напрямую).  ~5-8 минут на шесть пар.
# Требует DEEPSEEK_API_KEY.  Явный --model обязателен: дефолт раннера —
# openai/gpt-4.1-mini, а такого имени на api.deepseek.com нет.
run-llm-deepseek:
	ALL_PROXY= uv run -m src.runners.llm --deepseek --model deepseek-v4-flash

# LLM на локальной llama-swap (RTX 3090).  Медленно, последовательно.
# Требует запущенный llama-swap на http://127.0.0.1:12434.
# Явный --model обязателен: по умолчанию раннер попросит gpt-4.1-mini.
# Здесь Qwen3.5-0.8B; остальные локальные модели (Llama-3.2-1B-Instruct,
# gemma-4-E2B-it-Q6K и др.) перечислены в AGENTS.md и в конфигурации llama-swap.
run-llm-local:
	uv run -m src.runners.llm --local --model Qwen3.5-0.8B

# ── Сводная таблица ──────────────────────────────────────────────────────

# Печатает сводную таблицу из кэша results/.  Мгновенно.
compare:
	uv run -m src.runners.compare --table-only --xlsx

# Регенерирует results/comparison.xlsx (для ручного вызова после прогонов).
results/comparison.xlsx:
	uv run -m src.runners.compare --table-only --all-pairs --xlsx

# Запускает embedding + StringEquiv на всех 21 парах OAEI + печатает таблицу.
# LLM НЕ запускает (защита от случайных расходов).
compare-full:
	uv run -m src.runners.compare --run --all-pairs

# ── Абляции и проверки (без обращений к API) ────────────────────────────

# Абляция режима описания узла и правила отбора пар (MiniLM, ~45 с, CPU).
ablation:
	uv run python -m src.runners.embedding --modes --model MiniLM --direction canonical

# Потолок отбора кандидатов: recall@k для MiniLM, BM25 и гибридного ретривера.
retrieval:
	uv run python -m src.runners.retrieval

# Парные тесты между конфигурациями: Вилкоксон, доверительные интервалы,
# поправка Холма на двенадцать сравнений, macro и micro F1.
stats:
	uv run python -m src.runners.stats

# GNN: leave-one-pair-out по 21 паре (GPU, минуты).
gnn:
	uv run python -m src.runners.gnn --evaluate-all

# GNN: протокол с полностью отложенной онтологией (7 фолдов × 6 пар) —
# проверка переноса на невиданную онтологию, а не между парами.
gnn-ontology:
	uv run python -m src.runners.gnn --evaluate-all --fold-mode ontology-loo

# GNN: baseline по сырым признакам MiniLM под тем же правилом отбора 1:1.
gnn-baseline:
	uv run python -m src.runners.gnn --baseline

# Разброс генерации LLM: повторные прогоны одних пар (требует API-ключ, платно).
repeats:
	uv run python -m src.runners.repeats

# Проверка согласованности: целостность 7 онтологий OAEI + согласованность
# всех кэшированных маппингов, плюс эффект фильтра по флагам на F1.  Мгновенно.
consistency:
	uv run -m src.runners.consistency

# Провенанс: константы кода по каждому подходу рядом с числом записей,
# свежей датой mtime и суммарным числом API-запросов в `results/`.  Мгновенно,
# только чтение артефактов: результаты пишутся в `results/provenance.json`.
provenance:
	uv run python -m src.runners.provenance

# ── Документация ─────────────────────────────────────────────────────────

# Конвертирует note.md в Word (.docx) с сохранением формул LaTeX как OMML.
# Требуется pandoc ≥ 3.0.  Использует note-reference.docx для стилей (Times New Roman).
docx:
	pandoc note.md -o note.docx --reference-doc=note-reference.docx

# ── Визуализация ─────────────────────────────────────────────────────────

# Генерирует .dot в results/viz/: таксономии, все 21 пара (Ground Truth) и кэшированные маппинги.
viz:
	uv run -m src.viz.render

# Интерактивный просмотрщик таксономий и маппингов (tkinter).
viewer:
	uv run -m src.viz.viewer

# ── Проверка кода ────────────────────────────────────────────────────────

# Линтинг ruff (типы проверяются отдельным инструментом, если понадобится).
lint:
	uv run ruff check src/ tests/
	@# uv run mypy src/        # раскомментировать после настройки конфига.
	@# uv run pytest src/       # раскомментировать когда появятся тесты.

# ── Тесты ────────────────────────────────────────────────────────────────

# Юнит-тесты (pytest).  Без обращений к API, без обучения, без загрузки моделей.
test:
	uv run pytest

# ── Очистка ──────────────────────────────────────────────────────────────

# Удаляет все сгенерированные данные (data/processed/).
clean:
	rm -rf data/processed
	@echo "Cleaned data/processed."
