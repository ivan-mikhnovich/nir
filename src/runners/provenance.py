"""Провенанс результатов: что записано в коде и что реально лежит на диске.

Модуль ничего не пересчитывает и не ходит в сеть: он читает литералы из
исходников (через `ast`, без импорта раннеров, чтобы не мешать идущим
прогонам) и файлы кэша в `results/`, а затем показывает рядом две разные
вещи:

* `code_constant` — значение, взятое из литерала в коде, с указанием файла и
  символа в поле `origin`;
* `file_mtime` и `record_field` — число записей, крайние даты прогона и поля,
  реально записанные в JSON-записях кэша.

Разделение принципиально: большинство записей кэша не хранит ни модель, ни
гиперпараметры, поэтому «так настроен подход в коде» и «так лежит на диске» —
разные утверждения, и выдавать одно за другое нельзя.

Список подходов не зашит: это все подкаталоги `results/`, кроме служебных
`src.cache.NON_RESULT_DIRS` (там лежат производные артефакты, а не результаты
матчеров).  Новый подкаталог результатов попадает в отчёт автоматически.

Запуск:
    uv run python -m src.runners.provenance
"""

from __future__ import annotations

import argparse
import ast
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.cache import NON_RESULT_DIRS

RESULTS_DIR = Path("results")

# Поля, не попавшие в основные колонки, печатаются в колонке «прочее (код)».
EXTRA_SHOWN = 8

# Исходники, из которых читаются литералы конфигурации.
CODE_FILES: dict[str, Path] = {
    "compare": Path("src/runners/compare.py"),
    "embedding": Path("src/runners/embedding.py"),
    "gnn": Path("src/runners/gnn.py"),
    "llm_matcher": Path("src/matchers/llm.py"),
    "gnn_matcher": Path("src/matchers/gnn.py"),
}

# Известные виды подходов: каталог в results/ -> вид.  Вид выбирает только
# способ сбора констант кода; список подходов берётся из самого каталога,
# поэтому новый подкаталог результатов не теряется (вид «unknown»).
APPROACH_KIND: dict[str, str] = {
    "embedding-LaBSE": "embedding",
    "embedding-MiniLM": "embedding",
    "embedding-ruRoberta-large": "embedding",
    "embedding-rubert-base": "embedding",
    "string-equiv": "string",
    "gnn": "gnn",
    "gnn-baseline": "gnn-baseline",
    "llm-gpt": "llm",
    "llm-deepseek": "llm",
    "llm-bm25": "llm",
    "llm-hybrid": "llm",
}

# Функции-раннеры LLM-подходов в compare.py: подход -> имя функции.
LLM_RUNNERS: dict[str, str] = {
    "llm-gpt": "run_llm_gpt",
    "llm-deepseek": "run_llm_deepseek",
    "llm-hybrid": "run_llm_hybrid",
    "llm-bm25": "run_llm_bm25",
}

# Конструкторы матчеров: у их вызовов читаются именованные аргументы-литералы.
MATCHER_CALLS = frozenset(
    {"LLMMatcher", "BM25LLMMatcher", "HybridLLMMatcher", "EmbeddingMatcher"}
)

# Поля записей кэша, которые имеет смысл показать в разрезе «как на диске».
RECORD_FIELDS = (
    "approach",
    "mode",
    "model",
    "postprocess",
    "rule",
    "min_votes",
    "modes",
    "seed",
    "requests",
    "config",
)

# Порядок показа полей записи: сначала то, что отличает прогоны друг от друга.
DISPLAY_ORDER = (
    "config.fold_mode",
    "config.synthetic_seeds",
    "config.linear",
    "model",
    "postprocess",
    "rule",
    "seed",
    "requests",
    "modes",
    "min_votes",
    "mode",
    "approach",
)

# Сколько различных значений поля записи показывать в сводке.
MAX_FIELD_VALUES = 6

# Поля «код против диска» в порядке колонок Markdown-таблицы.
CODE_TABLE_FIELDS = (
    "model",
    "top_k",
    "temperature",
    "description_mode",
    "rule",
    "fold_mode",
    "seed",
    "epochs",
)


# ── Чтение литералов из исходников ──────────────────────────────────────


def parse_file(name: str) -> ast.Module:
    """Разобрать исходник в AST (без импорта самого модуля)."""
    return ast.parse(CODE_FILES[name].read_text(encoding="utf-8"))


def literal(node: ast.AST | None) -> Any:
    """Вернуть значение узла, если он литерал, иначе `None`."""
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def expression_value(node: ast.AST | None, consts: dict[str, Any]) -> Any:
    """Оценить выражение по литералам и константам модуля.

    Понимает литералы, имена модульных констант, `len(<имя>)` и f-строки из
    таких частей — этого достаточно для значений вида
    `f">={ENSEMBLE_MIN_VOTES}-of-{len(LLM_MODES)}+greedy_injective"`.
    """
    if node is None:
        return None
    value = literal(node)
    if value is not None:
        return value
    if isinstance(node, ast.Name):
        return consts.get(node.id)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Path"
        and len(node.args) == 1
    ):
        # `Path("results/gnn")` как константа каталога.
        return literal(node.args[0])
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "list"
        and len(node.args) == 1
    ):
        inner = expression_value(node.args[0], consts)
        return list(inner) if isinstance(inner, (list, tuple, set)) else None
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "len"
        and len(node.args) == 1
    ):
        inner = expression_value(node.args[0], consts)
        return len(inner) if isinstance(inner, (list, tuple, dict, str)) else None
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for piece in node.values:
            if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                parts.append(piece.value)
            elif isinstance(piece, ast.FormattedValue):
                inner = expression_value(piece.value, consts)
                if inner is None:
                    return None
                parts.append(str(inner))
            else:
                return None
        return "".join(parts)
    return None


def module_constants(name: str) -> dict[str, Any]:
    """Собрать модульные константы-литералы: имя -> значение."""
    consts: dict[str, Any] = {}
    for node in parse_file(name).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        else:
            continue
        if isinstance(target, ast.Name):
            value = expression_value(node.value, consts)
            if value is not None:
                consts[target.id] = value
    return consts


def function_defs(name: str) -> dict[str, ast.FunctionDef]:
    """Все функции верхнего уровня исходника: имя -> узел."""
    return {
        node.name: node
        for node in parse_file(name).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def class_defs(name: str) -> dict[str, ast.ClassDef]:
    """Все классы верхнего уровня исходника: имя -> узел."""
    return {
        node.name: node
        for node in parse_file(name).body
        if isinstance(node, ast.ClassDef)
    }


def call_keywords(
    name: str, func: str, callees: frozenset[str]
) -> dict[str, Any]:
    """Литеральные именованные аргументы вызовов `callees` внутри функции."""
    consts = module_constants(name)
    found: dict[str, Any] = {}
    node = function_defs(name).get(func)
    if node is None:
        return found
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        called = call.func.id if isinstance(call.func, ast.Name) else None
        if called in callees:
            for keyword in call.keywords:
                value = expression_value(keyword.value, consts)
                if keyword.arg and value is not None:
                    found[keyword.arg] = value
    return found


def constructor_defaults(name: str, cls: str) -> dict[str, Any]:
    """Значения по умолчанию параметров `cls.__init__` (только литералы)."""
    consts = module_constants(name)
    node = class_defs(name).get(cls)
    if node is None:
        return {}
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            args = item.args
            names = [a.arg for a in args.args + args.kwonlyargs]
            defaults = [None] * (len(names) - len(args.defaults)) + list(args.defaults)
            return {
                arg: expression_value(default, consts)
                for arg, default in zip(names, defaults)
                if expression_value(default, consts) is not None
            }
    return {}


def argparse_defaults(name: str, func: str = "main") -> dict[str, Any]:
    """Значения по умолчанию CLI-опций раннера: `--seed` -> значение."""
    consts = module_constants(name)
    found: dict[str, Any] = {}
    node = function_defs(name).get(func)
    if node is None:
        return found
    for call in ast.walk(node):
        if not isinstance(call, ast.Call) or not call.args:
            continue
        if not (isinstance(call.func, ast.Attribute) and call.func.attr == "add_argument"):
            continue
        flags = [a.value for a in call.args if isinstance(a, ast.Constant)]
        flag = next((f for f in flags if isinstance(f, str) and f.startswith("--")), None)
        if flag is None:
            continue
        keywords = {kw.arg: kw.value for kw in call.keywords if kw.arg}
        action = expression_value(keywords.get("action"), consts)
        default = expression_value(keywords.get("default"), consts)
        if default is not None:
            found[flag] = default
        elif action == "store_true":
            found[flag] = False
        elif action == "store_false":
            found[flag] = True
    return found


# ── Конфигурация подходов из кода ───────────────────────────────────────


def code_field(
    value: Any, origin: str, note: str | None = None
) -> dict[str, Any]:
    """Поле конфигурации с пометкой источника `code_constant`."""
    field: dict[str, Any] = {"value": value, "source": "code_constant", "origin": origin}
    if note:
        field["note"] = note
    return field


def embedding_code(approach: str) -> dict[str, Any]:
    """Константы эмбеддингового подхода."""
    short = approach.removeprefix("embedding-")
    consts = module_constants("compare")
    model_id = next(
        (mid for short_name, mid in consts.get("EMBEDDING_MODELS", []) if short_name == short),
        None,
    )
    # Режим описания и правило отбора берём из пересборки кэша.
    rebuild = call_keywords("embedding", "rebuild_caches", MATCHER_CALLS)
    postprocess = argparse_defaults("embedding").get("--postprocess")
    return {
        "model": code_field(model_id, "src/runners/compare.py:EMBEDDING_MODELS"),
        "description_mode": code_field(
            rebuild.get("description_mode"),
            "src/runners/embedding.py:rebuild_caches (EmbeddingMatcher)",
        ),
        "rule": code_field(
            postprocess, "src/runners/embedding.py:main --postprocess"
        ),
    }


def string_code() -> dict[str, Any]:
    """Константы StringEquiv: модели нет, правило отбора не применяется."""
    return {
        "model": code_field(
            None,
            "src/matchers/string_equiv.py:StringEquivMatcher",
            "лексическое совпадение имён, модель не используется.",
        ),
        "description_mode": code_field(
            "raw names (normalised)",
            "src/matchers/string_equiv.py:StringEquivMatcher.match",
            "сравниваются имена классов, приведённые к нижнему регистру.",
        ),
        "rule": code_field(
            "не более одного матча на источник (ничья — наименьший id цели)",
            "src/matchers/string_equiv.py:StringEquivMatcher.match",
        ),
    }


def gnn_code(approach: str) -> dict[str, Any]:
    """Константы GNN-протоколов из значений по умолчанию CLI."""
    options = argparse_defaults("gnn")
    consts = module_constants("gnn")
    rule_names = sorted(
        alias.name
        for node in parse_file("gnn").body
        if isinstance(node, ast.ImportFrom) and node.module == "matching_rules"
        for alias in node.names
        if alias.name in {"greedy_injective", "to_alignment"}
    )
    code = {
        "model": code_field(
            options.get("--embedder"),
            "src/runners/gnn.py:main --embedder",
            "признаки узла — эмбеддинг имени класса (name_only), путь передаётся графом.",
        ),
        "rule": code_field(
            " + ".join(rule_names) if rule_names else None,
            "src/runners/gnn.py:from ..matching_rules import",
        ),
        "description_mode": code_field(
            "name_only",
            "src/matchers/gnn.py:SiameseGraphSAGE._encode_taxonomy",
            "признак узла — эмбеддинг имени класса; структура передаётся матрицей смежности.",
        ),
        "fold_mode": code_field(options.get("--fold-mode"), "src/runners/gnn.py:main --fold-mode"),
        "seed": code_field(options.get("--seed"), "src/runners/gnn.py:main --seed"),
        "epochs": code_field(options.get("--epochs"), "src/runners/gnn.py:main --epochs"),
        "hidden_dim": code_field(options.get("--hidden-dim"), "src/runners/gnn.py:main --hidden-dim"),
        "out_dim": code_field(options.get("--out-dim"), "src/runners/gnn.py:main --out-dim"),
        "num_layers": code_field(options.get("--num-layers"), "src/runners/gnn.py:main --num-layers"),
        "dropout": code_field(options.get("--dropout"), "src/runners/gnn.py:main --dropout"),
        "lr": code_field(options.get("--lr"), "src/runners/gnn.py:main --lr"),
        "weight_decay": code_field(options.get("--weight-decay"), "src/runners/gnn.py:main --weight-decay"),
        "neg_ratio": code_field(options.get("--neg-ratio"), "src/runners/gnn.py:main --neg-ratio"),
        "synthetic_seeds": code_field(
            options.get("--synthetic-seeds"), "src/runners/gnn.py:main --synthetic-seeds"
        ),
        "linear": code_field(options.get("--linear"), "src/runners/gnn.py:main --linear"),
        "checkpoint_dir": code_field(
            str(consts.get("RESULTS_DIR")), "src/runners/gnn.py:RESULTS_DIR"
        ),
    }
    if approach == "gnn-baseline":
        code["epochs"] = code_field(0, "src/runners/gnn.py:_evaluate_baseline", "baseline не обучается.")
        code["fold_mode"] = code_field(
            None, "src/runners/gnn.py:_evaluate_baseline", "фолдов нет: оцениваются все 21 пара."
        )
    return code


def llm_code(approach: str) -> dict[str, Any]:
    """Константы LLM-подхода: вызов матчера, режимы, ансамбль, ретривер."""
    compare = module_constants("compare")
    matcher = module_constants("llm_matcher")
    kwargs = call_keywords("compare", LLM_RUNNERS[approach], MATCHER_CALLS)
    defaults = constructor_defaults("llm_matcher", "LLMMatcher")
    origin = f"src/runners/compare.py:{LLM_RUNNERS[approach]}"
    modes = compare.get("LLM_MODES") or list(matcher.get("ENSEMBLE_MODES", [])) or None
    min_votes = compare.get("ENSEMBLE_MIN_VOTES") or matcher.get("ENSEMBLE_MIN_VOTES")
    modes_origin = (
        "src/runners/compare.py:LLM_MODES"
        if compare.get("LLM_MODES")
        else "src/matchers/llm.py:ENSEMBLE_MODES"
    )
    vote_origin = (
        "src/runners/compare.py:ENSEMBLE_MIN_VOTES"
        if compare.get("ENSEMBLE_MIN_VOTES")
        else "src/matchers/llm.py:ENSEMBLE_MIN_VOTES"
    )
    ensemble = (
        f">={min_votes}-of-{len(modes)}+greedy_injective" if modes and min_votes else None
    )
    base_url = kwargs.get("base_url") or defaults.get("base_url")
    base_origin = (
        origin
        if kwargs.get("base_url")
        else "src/matchers/llm.py:LLMMatcher.__init__ (base_url, значение по умолчанию)"
    )
    retriever = (compare.get("RETRIEVER_KIND") or {}).get(approach)

    def field(key: str) -> dict[str, Any]:
        """Поле из вызова раннера, иначе — из значений по умолчанию конструктора."""
        explicit = kwargs.get(key)
        if explicit is not None:
            return code_field(explicit, f"{origin} ({key})")
        if defaults.get(key) is not None:
            return code_field(
                defaults[key],
                f"src/matchers/llm.py:LLMMatcher.__init__ ({key}, значение по умолчанию)",
            )
        return code_field(None, f"{origin} ({key})")

    return {
        "model": field("model"),
        "base_url": code_field(base_url, base_origin),
        "top_k": field("top_k"),
        "temperature": field("temperature"),
        "max_workers": field("max_workers"),
        "structured_output": field("use_structured_output"),
        "modes": code_field(modes, modes_origin),
        "rule": code_field(ensemble, "src/runners/compare.py:ENSEMBLE_RULE"),
        "ensemble": code_field(ensemble, "src/runners/compare.py:ENSEMBLE_RULE"),
        "confidence_threshold": code_field(
            matcher.get("CONFIDENCE_THRESHOLD"),
            "src/matchers/llm.py:CONFIDENCE_THRESHOLD",
        ),
        "min_votes": code_field(min_votes, vote_origin),
        "retriever": code_field(retriever, "src/runners/compare.py:RETRIEVER_KIND"),
    }


def discover_approaches(results_dir: Path = RESULTS_DIR) -> list[str]:
    """Найти подходы: подкаталоги `results/`, кроме служебных.

    Единственный источник правды о служебных каталогах —
    `src.cache.NON_RESULT_DIRS`, поэтому новый подкаталог результатов
    попадает в отчёт сам, без правки списка имён.
    """
    if not results_dir.exists():
        return []
    return sorted(
        path.name
        for path in results_dir.iterdir()
        if path.is_dir() and path.name not in NON_RESULT_DIRS
    )


def code_config(approach: str) -> dict[str, Any]:
    """Собрать конфигурацию подхода из кода по его виду."""
    kind = APPROACH_KIND.get(approach, "unknown")
    if kind == "embedding":
        return embedding_code(approach)
    if kind == "string":
        return string_code()
    if kind in {"gnn", "gnn-baseline"}:
        return gnn_code(approach)
    if kind == "llm":
        return llm_code(approach)
    # Неизвестный подкаталог: констант кода для него нет, но запись сохраняется.
    return {}


# ── Что лежит на диске ──────────────────────────────────────────────────


def jsonable(value: Any) -> Any:
    """Вернуть значение, если оно сериализуемо, иначе строку."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, (str, int, float, bool)) for item in value
    ):
        return list(value)
    return str(value)


def scan_disk(directory: Path) -> dict[str, Any]:
    """Записи каталога подхода: число, крайние даты, запросы, поля записей."""
    paths = sorted(directory.glob("*.json")) if directory.exists() else []
    records: list[tuple[Path, dict]] = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "pair" in data:
            records.append((path, data))
    mtimes = [path.stat().st_mtime for path, _ in records]
    requests = [
        record["requests"]
        for _, record in records
        if isinstance(record.get("requests"), (int, float))
    ]
    fields: dict[str, list[Any]] = {}
    for _, record in records:
        flat: dict[str, Any] = {}
        for key, value in record.items():
            flat[key] = value
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat[f"{key}.{subkey}"] = subvalue
        for key, value in flat.items():
            if key.split(".")[0] not in RECORD_FIELDS:
                continue
            seen = fields.setdefault(key, [])
            item = jsonable(value)
            if item not in seen and len(seen) < MAX_FIELD_VALUES:
                seen.append(item)
    return {
        "records": {"value": len(records), "source": "file_mtime"},
        "newest_mtime": {
            "value": stamp(max(mtimes)) if mtimes else None,
            "source": "file_mtime",
        },
        "oldest_mtime": {
            "value": stamp(min(mtimes)) if mtimes else None,
            "source": "file_mtime",
        },
        "api_requests": {
            "value": int(sum(requests)) if requests else None,
            "records_with_requests": len(requests),
            "source": "record_field",
        },
        "record_fields": {
            key: {"values": values, "source": "record_field"}
            for key, values in sorted(fields.items())
        },
    }


def stamp(epoch: float) -> str:
    """Локальная метка времени для mtime."""
    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")


# ── Отчёт ───────────────────────────────────────────────────────────────


def build_report(results_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    """Собрать отчёт: по каждому подходу — код рядом с состоянием на диске."""
    approaches: dict[str, Any] = {}
    for approach in discover_approaches(results_dir):
        approaches[approach] = {
            "kind": APPROACH_KIND.get(approach, "unknown"),
            "code": code_config(approach),
            "disk": scan_disk(results_dir / approach),
        }
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sources": {
            "code_constant": "значение взято из литерала исходника, файл и символ указаны в origin.",
            "file_mtime": "число записей и крайние даты изменения файлов results/<подход>/*.json.",
            "record_field": "поле, реально записанное в JSON-записях кэша (а не выведенное из кода).",
        },
        "code_files": {key: str(path) for key, path in CODE_FILES.items()},
        "non_result_dirs": sorted(NON_RESULT_DIRS),
        "approaches": approaches,
    }


def cell(field: dict[str, Any] | None) -> str:
    """Значение поля для Markdown-ячейки."""
    if not field:
        return "—"
    value = field.get("value")
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "да" if value else "нет"
    return str(value)


def extra_cell(code: dict[str, Any]) -> str:
    """Остальные константы подхода — одной ячейкой «ключ=значение»."""
    shown = [
        f"{key}={cell(field)}"
        for key, field in code.items()
        if key not in CODE_TABLE_FIELDS and field.get("value") is not None
    ]
    return "; ".join(shown[:EXTRA_SHOWN]) if shown else "—"


def display_fields(fields: dict[str, dict]) -> list[str]:
    """Ключи полей записей в порядке важности для различения прогонов."""
    order = {key: index for index, key in enumerate(DISPLAY_ORDER)}
    return sorted(fields, key=lambda key: (order.get(key, len(DISPLAY_ORDER)), key))


def print_markdown(report: dict[str, Any]) -> None:
    """Напечатать две таблицы: константы кода и состояние на диске."""
    print(f"Провенанс результатов, собрано {report['generated_at']}\n")
    print("### Константы кода (`source: code_constant`)\n")
    print("| подход | " + " | ".join(CODE_TABLE_FIELDS) + " | прочее (код) |")
    print("|" + "---|" * (len(CODE_TABLE_FIELDS) + 2))
    for approach, block in report["approaches"].items():
        code = block["code"]
        row = " | ".join(cell(code.get(f)) for f in CODE_TABLE_FIELDS)
        print(f"| {approach} | {row} | {extra_cell(code)} |")

    print("\n### Что лежит на диске (`source: file_mtime` / `record_field`)\n")
    print("| подход | записей | свежий mtime | старый mtime | API-запросов | поля в записях |")
    print("|---|---|---|---|---|---|")
    for approach, block in report["approaches"].items():
        disk = block["disk"]
        fields = disk["record_fields"]
        shown = ", ".join(
            f"{key}={fields[key]['values'][0]!r}" for key in display_fields(fields)[:4]
        )
        requests = disk["api_requests"]
        total = (
            "—" if requests["value"] is None
            else f"{requests['value']} ({requests['records_with_requests']})"
        )
        print(
            f"| {approach} | {disk['records']['value']} | {disk['newest_mtime']['value']} | "
            f"{disk['oldest_mtime']['value']} | {total} | {shown or '—'} |"
        )
    print("\n`API-запросов` — сумма поля `requests` по записям (в скобках — сколько записей его несут).")
    print("`поля в записях` — первые четыре поля в порядке важности; полный набор в JSON.")
    print("Константы кода в записях не хранятся: сравнивать нужно столбцы двух таблиц.")


def main() -> None:
    """Собрать провенанс, записать JSON и напечатать таблицы."""
    parser = argparse.ArgumentParser(
        description="Сводка происхождения результатов: константы кода против диска.",
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="каталог с кэшем подходов (по умолчанию results/).")
    parser.add_argument("--out", type=Path, default=None,
                        help="куда писать JSON (по умолчанию <results-dir>/provenance.json).")
    args = parser.parse_args()

    report = build_report(args.results_dir)
    out = args.out or args.results_dir / "provenance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_markdown(report)
    print(f"\nЗаписано в {out}")


if __name__ == "__main__":
    main()
