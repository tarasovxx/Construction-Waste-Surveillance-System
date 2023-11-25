# Система контроля за строительными отходами от команды Users

Мы, команда Users, разработали систему искусственного интеллекта, которая с помощью видеокамеры в режиме реального времени отслеживает подъезжающий самосвал и точно определяет к какому классу относится содержимое кузова. Взаимодействие с платформой осуществялется через удобный веб-сервис, который позволяет отслеживать работу в реальном времени и предоставляет вероятностным анализом содержимого кузова самосвала.



Мы используем современный подход детекции движения, чтобы точно сегментировать содержимое кузова подъехавшего самосвала на изображении и исключить влияние помех окружающего пространства, а после классифицируем класс объекта сверточной предобученной нейросетью.



Уникальность нашего решения заключается в его быстроте: оно способно лишь на одной CPU работать в режиме большой нагрузки и обрабатывать до 20 запросов в минуту, что позволяет Депстрою использовать его на практике.


Технические особенности:
Использование препроцессинга: цветовая фильтрация, стабилизация камеры, детекция движений, выделение контуров самосвала.
Многоуровневая классификация: ResNet (transfer learning)
- Эмоциональная окраска текста с помощью XLM Roberta, обученного на комментариях пользователей 
- Интерфейс на основе Streamlit


# Пример решения
<img width="1473" alt="image" src="https://github.com/tarasovxx/Construction-Waste-Surveillance-System/assets/42536677/18c5a224-7c63-4661-b056-7793210bd1e3">
<img width="1456" alt="image" src="https://github.com/tarasovxx/Construction-Waste-Surveillance-System/assets/42536677/b3f5dcfb-09d4-49ea-88fb-c8641a371670">


# Установка
- `git clone https://github.com/tarasovxx/Construction-Waste-Surveillance-System`
Необходим Python версии 3.9 и выше.
`pip install -r requirements.txt`
# Запуск
```bash
streamlit run app.py
```

# Используемое решение

Мы используем современный подход детекции движения, чтобы точно сегментировать содержимое кузова подъехавшего самосвала на изображении и исключить влияние помех окружающего пространства, а после классифицируем класс объекта сверточной предобученной нейросетью.


# Уникальность:

Наше решение отличается быстротой: оно способно лишь на одной CPU работать в режиме большой нагрузки и обрабатывать до 20 запросов в минуту, что позволяет Депстрою использовать его на практике.

# Стек используемых технологий:

`Python3`, `git`, `GitHub` - инструменты разработки

///////
`HF Transformers`, `TweetNLP`, `BertTopic` - библиотеки глубокого обучения

`Scikit-Learn`, `UMAP`, `KMeans` - фреймворки машинного обучения  
//////

`Plotly`, `Streamlit`, `SciPy` - инструменты визуализации  


# Сравнение моделей

//////
| Model  Description                                                | F1 Macro | Time    |
|--------------------------------------------------------|----------|---------|
| NaiveModel         каждое слово = новый кластер                                     | 0.81     | 10 ms   |
| LevensteinSimilarityModel    Если ответы схожие более, чем на 63% = образуют один кластер                          | 0.87     | 102 ms  |
| LevenshteinSimilatity + Processing Lemmatization, delete punct | 0.89     | 1 s     |
| SelfClusterModel#1 + SentimentTransformer (Bert-Multilingual + PCA + KMeans ) + (TweetNLP + xlm-roberta-multilingual) | 0.92     |         |
| SelfClusterModel#2                                     | 0.94     | 6 s     |
| SelfClusterModel#2 + SentimentTransformer              | 0.97     |         |
///////





# Проводимые исследования

- `notebook.ipynb` - исследования с детекцией объектов на видео и предобработка данных 

# Документация Django API

Этот проект использует Django и Django Rest Framework для создания API. API предоставляет доступ к информации о вопросах и ответах (QA) и содержит следующие эндпойнты:

### Эндпойнт `/api/process_video/{file_name}` Этот эндпойнт предоставляет доступ к загрузке видео на сервер. Он поддерживает следующие методы

- `POST`: Загрузка нового видео и получение адресов с обработанными данными.

### Структура данных

Структура базы данных включает в себя следующие поля

- `wood`: Вероятность, что в кузове дерево.
- `stone`: Вероятность, что в кузове кирпич.
- `concrete`: Вероятность, что в кузове бетон.
- `ground`: Вероятность, что в кузове грунт.

JSON имеет такой вид
```
{
"wood": 0.7
"stone": 0.1
"concrete": 0.1
"ground": 0.1
}
```

### Установка и запуск

Чтобы установить и запустить проект, выполните следующие шаги:
1. Клонируйте репозиторий с помощью `git clone`.
2. Создайте и активируйте виртуальное окружение.
3. Установите зависимости, выполнив команду `pip install -r requirements.txt`.
4. Примените миграции базы данных с помощью `python manage.py migrate`.
5. Запустите сервер с помощью `python manage.py runserver`.

## Примеры использования

Примеры запросов к API:
- Получение всех элементов QA: 
http://localhost:8000


# Разработчики
| Имя                  | Роль           | Контакт               |
|----------------------|----------------|-----------------------|
| Константин Балцат    | Data Analyse | [t.me/baltsat](https://t.me/baltsat)       |
| ---                  | ---            | ---                   |
| Александр Серов      | Machine Learning | [t.me/thegoldian](https://t.me/thegoldian) |
| ---                  | ---            | ---                   |
| Артем Тарасов        | Full stack | [t.me/tarasovxx](https://t.me/tarasovxx)   |
| ---                  | ---            | ---                   |
| Ванданов Сергей      | Machine Learning | [t.me/rapid76](https://t.me/@rapid76)      |
| ---                  | ---            | ---                   |
| Даниил Галимов       | Backend Developer | [t.me/Dan_Gan](https://t.me/Dan_Gan)  |
| ---                  | ---            | ---                   |



