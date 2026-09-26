"""Тесты поведения генератора провенанса: обнаружение подходов и дата прогона.

Оба теста работают на временных каталогах, ничего не читают из живого
`results/`, не ходят в сеть и не загружают модели.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.cache import NON_RESULT_DIRS
from src.runners.provenance import build_report, discover_approaches, scan_disk, stamp


def make_record(path: Path, stamp_value: float) -> None:
    """Записать минимальную запись кэша с заданной датой изменения."""
    path.write_text(json.dumps({"pair": "a↔b", "f1": 0.5}), encoding="utf-8")
    os.utime(path, (stamp_value, stamp_value))


def test_report_covers_exactly_the_result_subdirectories(tmp_path: Path) -> None:
    """Отчёт содержит ровно подкаталоги results/ минус служебные каталоги."""
    for name in ("embedding-LaBSE", "gnn", "llm-hybrid", "viz", "consistency", "brand-new"):
        (tmp_path / name).mkdir()
    # Файл на верхнем уровне — не подход, его учитывать нельзя.
    (tmp_path / "provenance.json").write_text("{}", encoding="utf-8")

    expected = {
        path.name
        for path in tmp_path.iterdir()
        if path.is_dir() and path.name not in NON_RESULT_DIRS
    }
    discovered = set(discover_approaches(tmp_path))

    assert discovered == expected
    # Служебные каталоги в отчёт не попадают, а новый подкаталог — попадает.
    assert discovered.isdisjoint(NON_RESULT_DIRS)
    assert "brand-new" in discovered
    assert set(build_report(tmp_path)["approaches"]) == expected


def test_newest_mtime_wins_over_oldest(tmp_path: Path) -> None:
    """Дата прогона — самая свежая запись, а не самая старая."""
    newest, oldest = 1_700_003_600.0, 1_700_000_000.0
    # Файл с самой свежей датой идёт первым по алфавиту: «последний файл» не подходит.
    make_record(tmp_path / "a.json", newest)
    make_record(tmp_path / "b.json", oldest)
    # Файл без поля `pair` — не запись и на дату влиять не должен.
    (tmp_path / "not-a-record.json").write_text(json.dumps({"summary": {}}), encoding="utf-8")
    os.utime(tmp_path / "not-a-record.json", (1_700_999_999.0, 1_700_999_999.0))

    disk = scan_disk(tmp_path)

    assert disk["records"]["value"] == 2
    assert disk["newest_mtime"]["value"] == stamp(newest)
    assert disk["oldest_mtime"]["value"] == stamp(oldest)
    assert disk["newest_mtime"]["value"] != disk["oldest_mtime"]["value"]
    assert disk["newest_mtime"]["source"] == "file_mtime"
