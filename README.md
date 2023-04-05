# Распределенная линейная регрессия

# Воспроизводимость прохождения тестов

Способы запуска тестов:
- `sbt test`&nbsp;&mdash; более быстрый способ,
- `docker-compose up --build`&nbsp;&mdash; более медленный, но более надежный способ на случай проблем с первым.

# Идеи доработки:
- "итерироваться", используя `streaming` c `broadcast`-ом обновленных значений параметров после `RDD.barrier`;
- использовать перечисленные в [An In-Depth Analysis of Distributed Training of
Deep Neural Networks][paper] способы обновления весов,

[paper]: https://ieeexplore.ieee.org/abstract/document/9460556/
