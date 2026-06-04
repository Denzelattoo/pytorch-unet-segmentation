# Архитектура U-Net для сегментации изображений на PyTorch

Мини-проект в рамках обучения DeepLearning. Cпроектирована и реализована классическая нейросеть U-Net на PyTorch.

## Структура проекта
- `src/dataset.py` — кастомный класс `Dataset` для загрузки картинок, масок и их трансформации с помощью `torchvision.transforms.v2`.
- `src/model.py` — архитектура сети UNet, включая блоки двойной свертки (`DoubleConv`) и сквозные связи (`Skip Connections / torch.cat`).
- `train.py` — главный скрипт управления, содержащий цикл обучения, оптимизатор Adam и функцию потерь BCEWithLogitsLoss.

## Запуск
Установите зависимости: `pip install torch torchvision Pillow`
Запустите скрипт: `python train.py`