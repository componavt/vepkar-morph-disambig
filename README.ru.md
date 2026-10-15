🌐 [Read in English](README.md)

# vepkar-morph-disambig

Проект посвящён контекстному ранжированию кандидатов морфологического
разбора слова в корпусе ВепКар: выбору словоформы и набора
грамматических признаков из предложенных вариантов.

В репозитории представлены три направления исследования:

- `src/t1_features/` — модели на признаках;
- `src/t2_context/` — контекстные модели;
- `src/t3_transfer/` — перенос обучения между языковыми разновидностями.

Общие алгоритмы находятся в `src/core/`, обработчики команд —
в `src/commands/`. Точка входа командной строки — `src/cli.py`.

## Подготовка окружения

Требуются Python 3.11 или новее и Git.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pandas zstandard scikit-learn scipy pytest
python -m pytest -q
```

## Работа с данными

Получить исходные данные из
[dictorpus-data](https://github.com/componavt/dictorpus-data/):

```bash
python src/cli.py fetch-data v2026.09
```

Проверить корпус, построить набор экземпляров и создать разбиение:

```bash
python src/cli.py inspect-data krl
python src/cli.py build-instances
python src/cli.py make-splits
```

Языковые коды: `vep` — вепсский язык; `krl`, `olo`, `lud` —
собственно карельское, ливвиковское и людиковское наречия
карельского языка соответственно.

Описание получаемого разбиения — в
[data/derived/README.ru.md](data/derived/README.ru.md).

## Оценка качества

Реализован частотный базовый метод: кандидаты ранжируются по частоте
экспертных разборов в обучающей части. Качество оценивается отдельно
на наборах `dev` и `test`.

Для частотного метода реализована диагностика на `dev`:
сопоставление с экспертными ответами, построчная таблица CSV
и краткая числовая сводка.

[Порядок запуска и вычисления метрик](docs/evaluation.ru.md).
[Результаты частотного метода](docs/frequency_baseline.ru.md).
