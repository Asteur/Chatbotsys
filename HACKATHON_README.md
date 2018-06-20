# Установка DeepPavlov

1. Создайте виртуальную среду Python 3.6. Например, так:
```
virtualenv -p python3 env
```
Но лучше conda:
```
conda create -n myenv python==3.6
```
2. Активируйте среду:
```
source ./env/bin/activate
```
Или, если это conda:
```
source activate myenv
```
3. Склонируйте ветку `odqa_hack` проекта **DeepPavlov**:
```
git clone -b odqa_hack https://github.com/deepmipt/DeepPavlov.git
```
4. Перейдите в корень проекта:
```
cd DeepPavlov
```
5. Установите нужные зависимости:
```
python setup.py
python -m spacy download en
```

## Запуск ODQA на инференс

1. Перейдите в deeppavlov:
```
cd deeppavlov/
```
2. Скачайте нужные данные с сервера:
```
python deep.py download configs/odqa/odqa_hack.json
```
3. Запустите процесс интерактивного общения c ODQA (**stdin-stdout**):
```
python hack.py
```
4. Задавайте ODQA вопросы на русском языке и наслаждайтесь жизнью.

5. Смотрите результаты бесед в файле `conversation_result.csv`

## Запуск процесса тренировки

1. Перейдите в deeppavlov:
```
cd deeppavlov/
```
2. Запустите процесс тренировки. В результате тренировки по указанному пути появятся
 файлы **data.db** и **tfidf.npz**
 ```
 python train.py {INPUT_FOLDER} {OUTPUT_FOLDER}
 ```
3. Тренировщик пересобирает БД и модель каждые 3 минуты.

## Запуск процесса инференса

1. Перейдите в deeppavlov:
```
cd deeppavlov/
```

2. Запустите процесс инференса:
```
python infer.py
```

##
Если что-то не получается, не отчаивайтесь и обратитесь к тому, кто это напрограммировал.
