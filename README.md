# Data Science — учебные проекты

Портфолио учебных проектов: анализ данных, статистика, SQL и классическое ML.
Стек: Python, pandas, NumPy, scikit-learn, CatBoost, SciPy / statsmodels, PostgreSQL, matplotlib / seaborn / plotly.

## Проекты

| № | Проект | Задача | Ключевые инструменты |
|---|--------|--------|----------------------|
| 1 | [Booking reviews](01_booking_reviews_rating) | Регрессия: предсказание оценки отеля (Kaggle-соревнование), MAPE ≈ 13.5 % на валидации | pandas, feature engineering, CatBoost, RandomForest |
| 2 | [Зарплаты в Data Science](02_ds_salaries_stat_tests) | Исследование + проверка гипотез статистическими тестами | SciPy, statsmodels, seaborn |
| 3 | [Вакансии HeadHunter (SQL)](03_hh_vacancies_sql) | Анализ базы вакансий SQL-запросами из Python | PostgreSQL, SQLAlchemy, BeautifulSoup |
| 4 | [Резюме HeadHunter (EDA)](04_hh_resume_eda) | Преобразование, очистка и разведывательный анализ 44 тыс. резюме | pandas, plotly |
| 5 | [Кредитный скоринг (ШИФТ 2025)](05_credit_scoring_shift) | Бинарная классификация дефолта, 1.2 млн строк — EDA и baseline, в доработке | pandas, scikit-learn |

## Школа 21

Учебные проекты ML-ветки Школы 21 — в папке [school21](school21): реализация алгоритмов «с нуля» и сравнение со scikit-learn.
Код в этих проектах написан с помощью AI-ассистента; подробнее — в README папки.

## Как запустить

В каждой папке лежит ноутбук и, где есть, `requirements.txt`. Датасеты в репозиторий не включены (кроме маленьких) — ссылки на источники указаны в README проектов; пути к данным в ноутбуках нужно заменить на свои.
