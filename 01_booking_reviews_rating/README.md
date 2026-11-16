# Booking reviews — предсказание оценки отеля

Учебное Kaggle-соревнование курса Skillfactory: по данным об отеле, авторе и метаданных отзыва предсказать оценку `reviewer_score`. Метрика — MAPE.

## Что сделано
- EDA: дубликаты, пропуски (координаты), выбросы, неинформативные признаки.
- Feature engineering: признаки из даты отзыва и `days_since_review`, национальность рецензента (топ стран + one-hot), логарифмирование скошенных признаков.
- Единая функция `preprocess_hotels` для train и test: границы выбросов и список топ-стран считаются только на train и переиспользуются на test.
- Модели: RandomForestRegressor (baseline) → CatBoostRegressor (`loss_function='MAPE'`).

## Результат
| Модель | MAPE на hold-out (20 %) |
|--------|------------------------|
| RandomForestRegressor | 0.1356 |
| CatBoostRegressor | 0.1353 |

## Данные
Датасет отзывов Booking.com (515K Hotel Reviews in Europe), файлы соревнования `hotels_train.csv` / `hotels_test.csv`. В репозиторий не включены — пути в ноутбуке нужно заменить на свои.

## Стек
Python 3.12, pandas, NumPy, scikit-learn, CatBoost, matplotlib, seaborn.
