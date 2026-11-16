# Школа 21 — ML-проекты

Учебные проекты ML-ветки Школы 21. Основной акцент — реализация алгоритмов «с нуля» на NumPy и сравнение с scikit-learn.

> Код в этих проектах написан с помощью AI-ассистента (Claude Code). Я использую их, чтобы разбирать устройство алгоритмов.

| Проект | Содержание |
|--------|------------|
| [ml2_linear_regression](ml2_linear_regression) | Линейная регрессия на SGD, Ridge / Lasso / ElasticNet, MinMaxScaler и StandardScaler, сравнение со sklearn. Данные: объявления об аренде (Renthop, Kaggle «Two Sigma Connect: Rental Listing Inquiries»). |
| [ml3_validation](ml3_validation) | Собственные разбиения (случайные и по дате), K-Fold / Group / Stratified / Time Series split, отбор признаков (Lasso, permutation importance, SHAP), Grid / Random Search и Optuna. |
| [ml4_classification](ml4_classification) | Соревнование «Don't Get Kicked!» (Carvana): собственные LogisticRegression, KNN, Naive Bayes, метрики Gini, Precision / Recall / F1, AUC PR, разбиение по времени. |
| [ml5_decision_trees](ml5_decision_trees) | Собственные DecisionTree (классификатор и регрессор), RandomForest, GBDT и ExtraTrees на данных «Don't Get Kicked!»; сравнение с LightGBM, CatBoost и XGBoost. Gini на валидации: своё дерево 0.44, свой лес 0.48, CatBoost 0.50. |
| [ml6_dimensionality_reduction](ml6_dimensionality_reduction) | PCA, SVD, t-SNE, UMAP, LLE на MNIST; модели на разреженных матрицах (Book Recommendation Dataset); сжатие изображений и выделение фона видео через SVD. |
| [movielens_analytics](movielens_analytics) | Модуль `movielens_analysis.py` (классы для фильмов, оценок, тегов, ссылок IMDB без pandas) и отчёт-ноутбук по датасету MovieLens. |

Датасеты в репозиторий не включены.
