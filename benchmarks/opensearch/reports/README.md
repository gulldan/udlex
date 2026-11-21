# Замеры OpenSearch 3.10.0

Один клиент отправлял локальные HTTP-запросы `_analyze` порциями до 256 словоформ. Узлу OpenSearch выделили четыре логических CPU, клиенту — два. Каждый прогон охватил 224 языка и пять конфигураций. Столбцы скорости показывают медиану входных словоформ в секунду по всем 224 языкам. [Исходники стенда для этих прогонов](https://github.com/gulldan/udlex/tree/18de28bac217e008cb2b0e5a665c8c427b9281f6/benchmarks/opensearch) · [Методика](../README.md)

## Плагин 0.1.0

| Машина | Время прогона | `standard`, форм/с | `standard → udlex → lowercase`, форм/с | Результаты |
| --- | ---: | ---: | ---: | --- |
| Linux AMD64, Ryzen 9 7950X | 16 мин 08 с | 659 308 | 565 142 | [Сводка](plugin-0.1.0/linux-amd64/summary.json) · [По языкам](plugin-0.1.0/linux-amd64/by-language.csv) · [Параметры запуска](plugin-0.1.0/linux-amd64/run.json) |
| Docker Desktop, Apple Silicon | 14 мин 46 с | 757 249 | 679 662 | [Сводка](plugin-0.1.0/linux-arm64/summary.json) · [По языкам](plugin-0.1.0/linux-arm64/by-language.csv) · [Параметры запуска](plugin-0.1.0/linux-arm64/run.json) |

## Плагин 0.1.1

Обе машины измеряли один ZIP с SHA-256 `85cba0176f085214b7db91020e60d94d8465170df18b9aa949411845e05744c8`. В каждом прогоне совпали результаты сверки 224 языковых строк и пяти строк сравнения токенизаторов. Эталонные пары для расчёта среднего F1 есть у 209 языков. [Исходники проверенной версии](https://github.com/gulldan/udlex/tree/5db45a7ed7aaec1bc26c1473055d105264ef1336/integrations/opensearch/plugin)

| Машина | Время прогона | `standard`, форм/с | `standard → udlex → lowercase`, форм/с | Результаты |
| --- | ---: | ---: | ---: | --- |
| Linux AMD64, Ryzen 9 7950X | 16 мин 36 с | 686 285 | 527 901 | [Сводка](plugin-0.1.1/linux-amd64/summary.json) · [По языкам](plugin-0.1.1/linux-amd64/by-language.csv) · [Параметры запуска](plugin-0.1.1/linux-amd64/run.json) |
| Docker Desktop, Apple Silicon | 15 мин 41 с | 708 064 | 649 049 | [Сводка](plugin-0.1.1/linux-arm64/summary.json) · [По языкам](plugin-0.1.1/linux-arm64/by-language.csv) · [Параметры запуска](plugin-0.1.1/linux-arm64/run.json) |

## Поиск на десяти языках в одном индексе

На обеих машинах короткий прогон записал десять документов в `udlex-demo`. Поиск по лемме вернул нужный документ для каждого из десяти полей. Запрос «котами» по полям `text.rus` и `text.ukr` нашёл оба документа. В ответах сохранены леммы и подсветка исходных слов. [AMD64](plugin-0.1.1/smoke/linux-amd64/multilingual-search-smoke.json) · [ARM64](plugin-0.1.1/smoke/linux-arm64/multilingual-search-smoke.json) · [Запросы для Dev Tools](../../../integrations/opensearch/docker/example.http)
