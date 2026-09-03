---
tags: [quickstart, vision, fds]
dataset: [CIFAR-10, HAM10000, FER2013, Cassava Leaf Disease 2020]
framework: [torch, torchvision]
---

# Федеративное обучение с PyTorch и Flower (пример быстрого старта)

В этом вводном примере для Flower используется PyTorch, однако для его запуска не обязательно глубоко знать PyTorch. Тем не менее, это поможет понять, как адаптировать Flower для вашего сценария использования. Сам по себе запуск этого примера довольно прост. В нём применяются [Flower Datasets](https://flower.ai/docs/datasets/) для загрузки, разбиения и предварительной обработки набора данных CIFAR-10.

## Набор данных и non-IID-сценарии

В эксперименте используется CIFAR-10 с настраиваемым IID-разбиением или
разбиением Дирихле по меткам. По умолчанию выбран умеренно неоднородный non-IID-сценарий
`dirichlet-alpha=0.5`. Официальная тестовая выборка остаётся централизованной.
Обоснование выбора, ограничения и протокол эксперимента приведены в [DATASET.md](DATASET.md).

В экспериментах с естественным дисбалансом используется HAM10000 из Kaggle. Для его
семи классов соотношение большинства к меньшинству составляет примерно 58:1. Конвейер
хранит все изображения с одним `lesion_id` вместе, поддерживает естественную федерацию
из четырёх источников и при необходимости использует глобально сбалансированные веса
перекрёстной энтропии. См. [HAM10000.md](HAM10000.md).

Шесть наборов данных с естественным дисбалансом были оценены по прозрачной взвешенной
рубрике. Для HAM10000, FER2013 и Cassava 2020 имеются исполняемые адаптеры и
проверки работоспособности; научное сравнение и обоснование выбора приведены в
[DATASET_COMPARISON.md](DATASET_COMPARISON.md).

Загрузите и закэшируйте набор данных, затем сгенерируйте примеры изображений, таблицы классов
для каждого клиента и тепловые карты распределений:

```bash
python scripts/prepare_cifar10.py
```

Подготовьте HAM10000 (примерно 3,2 ГБ в кэше Kaggle):

```bash
python scripts/prepare_ham10000.py
```

Подготовьте лёгкий кандидат FER2013 (около 60 МБ):

```bash
python scripts/prepare_candidate_dataset.py --dataset fer2013
```

Подготовьте Cassava 2020 (около 6,19 ГБ; требуются аутентификация в Kaggle и
принятие правил соревнования):

```bash
python scripts/prepare_candidate_dataset.py --dataset cassava
```

Та же команда принимает `--source-dir` для уже существующей локальной загрузки.
Примеры переопределений Flower хранятся в `configs/fer2013_dirichlet.toml` и
`configs/cassava_dirichlet.toml`.

```bash
flwr run . --run-config configs/fer2013_dirichlet.toml --stream
flwr run . --run-config configs/cassava_dirichlet.toml --stream
```

## Настройка проекта

### Получение приложения

Установите Flower:

```shell
pip install flwr
```

Получите приложение:

```shell
flwr new @flwrlabs/quickstart-pytorch
```

Будет создан новый каталог `quickstart-pytorch` со следующей структурой:

```shell
quickstart-pytorch
├── pytorchexample
│   ├── __init__.py
│   ├── client_app.py   # Определяет ваш ClientApp
│   ├── server_app.py   # Определяет ваш ServerApp
│   └── task.py         # Определяет модель, обучение и загрузку данных
├── pyproject.toml      # Метаданные проекта: зависимости и конфигурации
└── README.md
```

### Установка зависимостей и проекта

Установите зависимости из `pyproject.toml`, а также пакет `pytorchexample`.

```bash
pip install -e .
```

## Запуск проекта

Проект Flower можно запускать как в режиме _симуляции_, так и в режиме _развёртывания_, не изменяя код. Если вы только начинаете работать с Flower, рекомендуем режим симуляции, поскольку в нём нужно вручную запускать меньше компонентов. По умолчанию `flwr run` использует Simulation Engine.

### Запуск с Simulation Engine

> [!TIP]
> Этот пример работает быстрее, когда у `ClientApp` есть доступ к GPU. Подробнее о симуляциях Flower и их оптимизации см. в [документации Simulation Engine](https://flower.ai/docs/framework/how-to-run-simulations.html).

```bash
# Запуск с федерацией по умолчанию (только CPU)
flwr run .  --stream
```

Можно также переопределить некоторые настройки `ClientApp` и `ServerApp`, заданные в `pyproject.toml`. Например:

```bash
flwr run . --run-config "num-server-rounds=5 learning-rate=0.05"  --stream
```

> [!TIP]
> Более подробное руководство см. в нашем [учебнике по быстрому старту с PyTorch](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html).

### Запуск с Deployment Engine

Следуйте этому [практическому руководству](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html), чтобы запустить то же приложение из этого примера с Deployment Engine Flower. Затем можно настроить в своей федерации [защищённую связь с TLS](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) и [аутентификацию SuperNode](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html).

Если вы уже знакомы с работой Deployment Engine, возможно, вам будет полезно узнать, как запускать его с помощью Docker. См. документацию [Flower with Docker](https://flower.ai/docs/framework/docker/index.html).
