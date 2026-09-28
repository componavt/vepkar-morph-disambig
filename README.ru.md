🌐 [Read in English](README.md)

# vepkar-morph-disambig

Проект посвящён контекстному ранжированию кандидатов морфологического
разбора слова в корпусе ВепКар: выбору словоформы и набора
грамматических признаков из предложенных вариантов.

В репозитории представлены три направления исследования:

- `src/t1_features/` — модели на признаках;
- `src/t2_context/` — контекстные модели;
- `src/t3_transfer/` — перенос обучения между языковыми разновидностями.

Общий код находится в `src/core/`.

## Подготовка окружения

Требуются Python 3.11 или новее и Git.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pandas zstandard pytest
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

Языковые коды: `vep` — вепсский, `krl` — собственно карельский,
`olo` — ливвиковский, `lud` — людиковский.

Описание получаемого разбиения — в
[data/derived/README.ru.md](data/derived/README.ru.md).