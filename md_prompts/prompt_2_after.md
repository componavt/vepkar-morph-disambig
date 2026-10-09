# Prompt 

На основе следующего метазапроса был построен и успешно выполнен prompt ИИ-агентом:
```
```

# GitHub

ИИ-агент выполнил промпт для решения задачи НАЗВАНИЕ ЗАДАЧИ. 

Промпт выполнен, код репозитория GitHub (https://github.com/componavt/vepkar-morph-disambig) обновлён. Прочитай и обнови своё представление об этом репозитории.  Прочитай обновлённую версию репозитория через GitHub-коннектор, либо сообщи о невозможности прочтения.

Вот output в VS Code после реализации ИИ-агентом промпта:
```
```

Вот результаты выполнения всего того, что указано в README.md в консоли:
```
```

Какие видишь ошибки в коде и как предлагаешь их исправлять?
Какие тесты стоит прогнать или какие скрипты в этом репозитории запустить, чтобы предъявить тебе результаты в консоли или фрагменты получившихся CSV-файлов, чтобы ты получил более полную картину? 

============================================================

Покажи и объясни мне текущее состояние кода с использованием ASCII art и эмодзи. 

Также перечисли какие группы задач остались.



============================================================
TOOL-CALL DISCIPLINE
============================================================

- Use the editor's native structured tool interface; do not print pseudo-tool calls,
  XML tool calls, Markdown JSON blocks, or narration instead of invoking a tool.
- For each edit, make one small atomic replacement in one full-path file.
- Before editing, read the exact target fragment from the file.
- The edit must send all required fields in the tool schema:
  filePath, oldString, newString.
- Use camelCase schema names exactly; do not use file_path, old_string, or new_string.
- If a tool validation error occurs, retry once with the exact required schema.
  Do not repeat narration or issue another empty tool call.
- Prefer several small edits over one large write containing a whole long source file.

===============

===============

===============

============================================================
ОГРАНИЧЕНИЕ ВЫВОДА И ФИНАЛЬНОГО ОТЧЁТА
============================================================

Экономь output tokens. Не пересказывай постановку задачи, не описывай ход
размышлений, не показывай успешные промежуточные tool calls и не повторяй
полные фрагменты изменённого кода.

После завершения дай краткий final report не более 900 токенов и только в
следующем формате:

1. Changed files
   - Полные относительные пути только реально изменённых файлов.

2. Completed
   - 3–8 коротких пунктов только о фактически реализованном поведении и
     важных schema/API изменениях.

3. Validation
   - Точные запущенные команды и только итог каждой команды:
     passed/failed/skipped counts.
   - Не вставляй полный успешный pytest output.

4. Problems / limitations
   - Пиши только если есть незавершённость, failure, отклонение от scope или
     значимое известное ограничение.
   - Для failure укажи точный test/command и одну краткую root cause.
   - Не называй задачу completed, если relevant tests не прошли.

Не сообщай о добавлении отдельной строки, импорта, метода или константы, если
это не меняет наблюдаемое поведение. Не повторяй требования prompt в отчёте.
