---
tags: [quickstart, vision, fds]
dataset: [CIFAR-10, HAM10000, FER2013, Cassava Leaf Disease 2020]
framework: [torch, torchvision]
---

# Федеративное обучение с PyTorch и Flower

Пример локальной федеративной симуляции для классификации изображений. Проект
поддерживает CIFAR-10, HAM10000, FER2013 и Cassava Leaf Disease 2020, стратегии
FedAvg, FedAvgM, FedProx и FedAdam, несколько сценариев разбиения данных между
клиентами и расширенные метрики для несбалансированной классификации.

## Быстрый запуск

Все команды выполняются из каталога `quickstart-pytorch`. Зависимости объявлены
в `pyproject.toml`, поэтому отдельный `requirements.txt` не нужен.

Требуется Python 3.10 или новее. Проверенный вариант — Python 3.13.

```bash
python3 -m venv venv
source venv/bin/activate

command -v python
python --version
python -m pip --version

python -m pip install --upgrade pip
python -m pip install -e .
```

После активации `command -v python` должен указывать на
`quickstart-pytorch/venv/bin/python`. Если вместо этого выводится строка вида
`alias python=/usr/bin/python3`, alias перекрывает виртуальное окружение:

```bash
unalias python
rehash
```

Затем удалите или закомментируйте соответствующий `alias python=...` в
`~/.zshrc`, чтобы проблема не повторялась в новых терминалах. Не используйте
для установки `sudo` или `--user`.

### Записываемый эксперимент

Основной способ запускать воспроизводимые эксперименты — через обёртку
`scripts/run_experiment.py`. Она создаёт каталог результата, передаёт Flower
эффективную конфигурацию и сохраняет манифест, окружение, консольный лог,
сырые таблицы метрик и отчёт.

```bash
python scripts/run_experiment.py \
  --config configs/ham10000_dirichlet.toml \
  --name baseline \
  --num-clients 10 \
  --rounds 20 \
  --learning-rate 0.01 \
  --save-model
```

Для этой команды результат записывается в каталог вида
`results/ham10000/YYYYMMDD-HHMMSS_baseline/`:

```text
results/
└── ham10000/
    └── YYYYMMDD-HHMMSS_baseline/
        ├── experiment.json
        ├── environment.json
        ├── console.log
        ├── round_metrics.csv
        ├── client_metrics.csv
        ├── per_class_metrics.csv
        ├── client_confusion_matrices.jsonl
        ├── final_model.pt
        ├── summary.md
        ├── confusion_matrices/
        │   ├── centralized_round_000.csv
        │   ├── centralized_round_001.csv
        │   └── federated_round_001.csv
        └── plots/
            ├── learning_curves.png
            ├── learning_curves.pdf
            ├── client_dispersion.png
            ├── client_dispersion.pdf
            ├── final_per_class_metrics.png
            ├── final_per_class_metrics.pdf
            ├── final_confusion_matrix.png
            ├── final_confusion_matrix.pdf
            ├── client_class_distribution.png
            └── client_class_distribution.pdf
```

Некоторые файлы появляются только если соответствующие данные были записаны:
например, `final_model.pt` создаётся при `--save-model`, клиентские таблицы —
после клиентской оценки, а имена CSV в `confusion_matrices/` зависят от
раундов и источников метрик.

Порядок применения конфигурации: значения по умолчанию из
`pyproject.toml` < TOML-файл из `--config` < параметры командной строки.
Ключ `experiment-dir` зарезервирован за обёрткой: в `pyproject.toml` он
объявлен пустым, а перед запуском заменяется абсолютным путём созданного
каталога результата.

Имя стратегии и только влияющие на неё параметры сохраняются в полях
`strategy` и `strategy_config` файла `experiment.json`. Если `--name` не задан,
стратегия также включается в автоматически создаваемое имя каталога.

Статусы в `experiment.json`:

- `running` — каталог создан, запуск ещё выполняется;
- `completed` — обучение и построение отчёта завершились без предупреждений;
- `completed_with_warnings` — основные сырые артефакты сохранены, но часть
  графиков или `summary.md` не удалось построить;
- `failed` — процесс Flower завершился с ненулевым кодом;
- `aborted` — запуск был прерван, например `Ctrl+C`.

Сырые файлы не удаляются при ошибках обучения или рендеринга отчёта. Если
Flower упал, остаются `experiment.json`, `environment.json` и `console.log`.
Если не построились графики или `summary.md`, уже записанные CSV, JSONL,
матрицы и модель остаются в каталоге эксперимента, а предупреждения
сохраняются в манифесте.

### CIFAR-10

По умолчанию приложение загружает CIFAR-10 (`uoft-cs/cifar10`) через Hugging
Face Datasets и запускает локальную Flower-симуляцию:

```bash
flwr run . --stream
```

Первый запуск может занять больше времени из-за загрузки и кэширования набора
данных.

### HAM10000

Подготовьте HAM10000 (примерно 3,2 ГБ в кэше Kaggle):

```bash
python scripts/prepare_ham10000.py
```

Скрипт создаёт локальные CSV-манифесты в `data/ham10000` и диагностические
таблицы и изображения в `dataset_examples/ham10000`. Возможны два ожидаемых
предупреждения:

- Kaggle-зеркало может не содержать официальное тестовое изображение
  `ISIC_0035068`; тогда тестовая выборка содержит 1511 изображений вместо 1512,
  а обучающая выборка остаётся полной;
- `DirichletPartitioner` может повторить построение диагностического разбиения
  с `alpha=0.1`, пока каждый клиент не получит минимум 50 примеров. Основной
  запуск ниже использует `alpha=0.5`.

Flower собирает приложение в FAB и запускает его копию из `~/.flwr/apps`.
Каталог `data/` в FAB не включается, поэтому для локального датасета необходимо
передать абсолютный путь:

```bash
flwr run . \
  --run-config "dataset='ham10000' dataset-root='$PWD/data/ham10000' partitioner='dirichlet' class-weighting='balanced'" \
  --stream
```

В Flower 1.31 TOML-файл нельзя передать одновременно с дополнительным
`--run-config`. Поэтому команда выше заменяет
`configs/ham10000_dirichlet.toml`, сохраняя его настройки, но подставляя
абсолютный `dataset-root`.

### FER2013 и Cassava 2020

Подготовьте локальный набор данных:

```bash
python scripts/prepare_candidate_dataset.py --dataset fer2013
python scripts/prepare_candidate_dataset.py --dataset cassava
```

Cassava 2020 занимает около 6,19 ГБ; для него требуются аутентификация в Kaggle
и принятие правил соревнования. Обе команды также принимают `--source-dir` для
уже существующей локальной загрузки.

Запускайте локальные датасеты с абсолютным путём по той же причине, что и
HAM10000:

```bash
flwr run . \
  --run-config "dataset='fer2013' dataset-root='$PWD/data/fer2013' partitioner='dirichlet' class-weighting='balanced'" \
  --stream

flwr run . \
  --run-config "dataset='cassava' dataset-root='$PWD/data/cassava' partitioner='dirichlet' class-weighting='balanced'" \
  --stream
```

## Что происходит во время запуска

Конфигурация по умолчанию создаёт 10 виртуальных клиентов и выполняет три
раунда FedAvg. Обучающая выборка делится между клиентами по меткам с помощью
распределения Дирихле (`alpha=0.5`, `seed=42`, минимум 50 примеров на клиента),
после чего локальная выборка каждого клиента делится на 80% для обучения и 20%
для валидации.

В каждом раунде все клиенты получают текущую CNN, обучают её одну локальную
эпоху с SGD (`learning-rate=0.1`, `momentum=0.9`, `batch-size=32`) и возвращают
веса серверу. Сервер усредняет их пропорционально числу обучающих примеров.
До первого раунда и после каждого раунда модель также оценивается на
централизованной тестовой выборке. В логах выводятся loss, accuracy, balanced
accuracy, macro/weighted precision, recall и F1, метрики каждого класса и
confusion matrix.

По умолчанию итоговые веса не сохраняются (`save-model=false`). Для запуска
CIFAR-10 с сохранением `final_model.pt` используйте:

```bash
flwr run . --run-config "save-model=true" --stream
```

Другие существующие параметры можно переопределить аналогично:

```bash
flwr run . \
  --run-config "num-server-rounds=5 learning-rate=0.05" \
  --stream
```

## Стратегии агрегации

По умолчанию используется `FedAvg`. Обёртка эксперимента поддерживает четыре
стратегии:

| Значение `strategy` | Алгоритм | Клиентское обучение |
| --- | --- | --- |
| `fedavg` | взвешенное FedAvg | стандартный SGD |
| `fedavgm` | FedAvg с серверным momentum | стандартный SGD |
| `fedprox` | FedAvg с проксимальным ограничением | SGD с proximal loss |
| `fedadam` | адаптивная серверная оптимизация Adam | стандартный SGD |

Примеры запуска на одном и том же разбиении HAM10000:

```bash
python scripts/run_experiment.py \
  --config configs/ham10000_dirichlet.toml \
  --num-clients 10 --rounds 20 --strategy fedavg

python scripts/run_experiment.py \
  --config configs/ham10000_dirichlet.toml \
  --num-clients 10 --rounds 20 --strategy fedprox --proximal-mu 0.01

python scripts/run_experiment.py \
  --config configs/ham10000_dirichlet.toml \
  --num-clients 10 --rounds 20 --strategy fedavgm \
  --server-learning-rate 1.0 --server-momentum 0.9

python scripts/run_experiment.py \
  --config configs/ham10000_dirichlet.toml \
  --num-clients 10 --rounds 20 --strategy fedadam \
  --fedopt-eta 0.1 --fedopt-beta-1 0.9 --fedopt-beta-2 0.99 \
  --fedopt-tau 0.001
```

Для FedAdam клиентский параметр `eta_l` автоматически равен
`learning-rate`, то есть соответствует фактическому локальному SGD. Параметры
выбранной стратегии проверяются до создания каталога результатов.
Во всех стратегиях `train_loss` означает сопоставимую между запусками
кросс-энтропию. Для FedProx полный оптимизируемый loss и величина
проксимального штрафа дополнительно записываются как `objective_loss` и
`regularization_loss`.

Для добавления серверной стратегии зарегистрируйте новую
`StrategyDefinition` в `pytorchexample/strategies.py`. Если алгоритм меняет
локальное обучение, добавьте реализацию `LocalTrainingAlgorithm` в
`pytorchexample/local_training.py` и укажите её имя в определении стратегии.
Серверное приложение, запись метрик и клиентская оркестрация при этом не
изменяются.

## Наборы данных и non-IID-сценарии

Для CIFAR-10 доступны IID-разбиение и разбиение Дирихле по меткам. По умолчанию
выбран умеренно неоднородный non-IID-сценарий `dirichlet-alpha=0.5`.
Официальная тестовая выборка остаётся централизованной. Обоснование выбора,
ограничения и протокол эксперимента приведены в [DATASET.md](DATASET.md).

В экспериментах с естественным дисбалансом используется HAM10000. Для его семи
классов соотношение большинства к меньшинству составляет примерно 58:1.
Конвейер хранит все изображения с одним `lesion_id` вместе, поддерживает
естественную федерацию из четырёх источников и при необходимости использует
глобально сбалансированные веса перекрёстной энтропии. Подробнее см.
[HAM10000.md](HAM10000.md).

Шесть наборов данных с естественным дисбалансом оценены по прозрачной
взвешенной рубрике. Для HAM10000, FER2013 и Cassava 2020 имеются исполняемые
адаптеры и проверки работоспособности. Научное сравнение и обоснование выбора
приведены в [DATASET_COMPARISON.md](DATASET_COMPARISON.md).

Для CIFAR-10 можно отдельно сгенерировать примеры изображений, таблицы классов
клиентов и тепловые карты распределений:

```bash
python scripts/prepare_cifar10.py
```

Готовые варианты конфигурации находятся в каталоге `configs/`. При запуске
локальных наборов учитывайте описанное выше ограничение относительных путей в
FAB.

## Устранение проблем

### `Defaulting to user installation` и несовместимая версия Python

Проверьте, что команда `python` действительно принадлежит окружению:

```bash
command -v python
python --version
python -m pip --version
```

При необходимости обойдите shell-alias и вызовите интерпретатор напрямую:

```bash
./venv/bin/python -m pip install -e .
./venv/bin/flwr run . --stream
```

### `Can't locate revision identified by ...`

Такое сообщение означает, что служебная база локального SuperLink была создана
несовместимой версией Flower. Сохраните её как резервную копию; следующий запуск
создаст новую базу:

```bash
mkdir -p ~/.flwr/local-superlink/backup
mv ~/.flwr/local-superlink/state.db ~/.flwr/local-superlink/backup/
mv ~/.flwr/local-superlink/state.db-shm ~/.flwr/local-superlink/backup/ 2>/dev/null
mv ~/.flwr/local-superlink/state.db-wal ~/.flwr/local-superlink/backup/ 2>/dev/null

flwr run . --stream
```

### `Missing ~/.flwr/apps/.../data/<dataset>/test.csv`

Данные существуют в рабочем каталоге, но не входят в FAB. Не копируйте их в
`~/.flwr/apps`: хеш каталога меняется при сборке. Запустите приложение с
абсолютным `dataset-root`, как показано в командах для HAM10000, FER2013 и
Cassava выше.

## Deployment Engine

Проект можно запускать не только в режиме симуляции, но и через Deployment
Engine без изменения кода. Инструкции по развёртыванию, TLS, аутентификации
SuperNode и Docker приведены в документации Flower:

- [Deployment Engine](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html)
- [TLS](https://flower.ai/docs/framework/how-to-enable-tls-connections.html)
- [аутентификация SuperNode](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html)
- [Flower with Docker](https://flower.ai/docs/framework/docker/index.html)

Дополнительный материал: [учебник по быстрому старту с
PyTorch](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html).
