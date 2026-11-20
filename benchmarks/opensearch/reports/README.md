# Замеры OpenSearch 3.10.0 от 20 ноября 2026 года

Один клиент отправлял локальные HTTP-запросы `_analyze` порциями до 256 словоформ. Узлу OpenSearch выделили четыре логических CPU, клиенту — два. Каждый прогон охватил 224 языка и пять конфигураций. Столбцы скорости показывают медиану входных словоформ в секунду по всем 224 языкам. [Исходники стенда для этих прогонов](https://github.com/gulldan/udlex/tree/18de28bac217e008cb2b0e5a665c8c427b9281f6/benchmarks/opensearch) · [Методика](../README.md)

| Машина | Время прогона | `standard`, форм/с | `standard → udlex → lowercase`, форм/с | Результаты |
| --- | ---: | ---: | ---: | --- |
| Linux AMD64, Ryzen 9 7950X | 16 мин 08 с | 659 308 | 565 142 | [Сводка](2026-11-20/linux-amd64/summary.json) · [По языкам](2026-11-20/linux-amd64/by-language.csv) · [Параметры запуска](2026-11-20/linux-amd64/run.json) |
| Docker Desktop, Apple Silicon | 14 мин 46 с | 757 249 | 679 662 | [Сводка](2026-11-20/linux-arm64/summary.json) · [По языкам](2026-11-20/linux-arm64/by-language.csv) · [Параметры запуска](2026-11-20/linux-arm64/run.json) |
