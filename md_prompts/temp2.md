iw

```
T1.1a
  pure feature/example construction
  + автономные tests
        │
        ▼
T1.1b
  train
  + model bundle persistence
        │
        ▼
T1.1c
  dev predictions CLI
  + existing validation/evaluation
        │
        ▼
T1.1d
  C = 0.1 / 1 / 10
  + selection on dev
  + result bundle
```

```
T1.1a
│
├─ a1  «Признаки»       Что известно об одном кандидате?
│          │
├─ a2  «Контекст»       Какие слова стоят слева и справа?
│          │
├─ a3  «Сборка»         Как получить CandidateExample?
│          │
└─ a4  «Приёмка»        Весь контракт действительно соблюдён?
           │
           ▼
         T1.1b — обучение
```
T1.1a1 — типы и построение признаков одного кандидата
T1.1a2 — разрешение контекста и его ошибки
T1.1a3 — сборка CandidateExample и режимы train/predict
T1.1a4 — приёмка полного контракта



