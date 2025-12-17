import sys
import os
import pandas as pd
import numpy as np

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import DataPreprocessor
from src.eda_visualization import EDAAnalyzer
from src.model_training import ModelTrainer


def main():
    """Основная функция выполнения анализа"""
    print("=" * 70)
    print("АНАЛИЗ ДАННЫХ О ЗДОРОВЬЕ СНА И ОБРАЗЕ ЖИЗНИ")
    print("=" * 70)

    # Конфигурация
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "Sleep_health_and_lifestyle_dataset.csv")
    PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed_sleep_health_data.csv")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")

    # ШАГ 1: Предобработка данных
    print("\n" + "=" * 70)
    print("ШАГ 1: ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ")
    print("=" * 70)

    preprocessor = DataPreprocessor(DATA_PATH)
    df_raw = preprocessor.load_data()

    # Анализ исходных данных
    missing_table = preprocessor.explore_data()

    # Очистка и преобразование
    df_clean = preprocessor.clean_data()

    # Подготовка признаков и целевой переменной
    X, y = preprocessor.prepare_features_target()

    # Получение списков признаков
    numerical_cols, categorical_cols = preprocessor.get_numerical_categorical_cols(X)

    # Сохранение обработанных данных
    preprocessor.save_processed_data(PROCESSED_DATA_PATH)

    # ШАГ 2: Разведочный анализ данных (EDA)
    print("\n" + "=" * 70)
    print("ШАГ 2: РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ (EDA)")
    print("=" * 70)

    analyzer = EDAAnalyzer()

    # 2.1 Распределение целевой переменной
    print("\nВизуализация распределения целевой переменной...")
    analyzer.plot_target_distribution(y, preprocessor.label_encoders['Sleep Disorder'])

    # 2.2 Распределение числовых признаков
    print("\nАнализ распределения числовых признаков...")
    analyzer.plot_numerical_distributions(df_clean, numerical_cols[:9], n_cols=3)

    # 2.3 Анализ выбросов
    print("\nАнализ выбросов в данных...")
    analyzer.plot_boxplots_outliers(df_clean, numerical_cols[:9], n_cols=3)

    # 2.4 Корреляционный анализ
    print("\nАнализ корреляций между признаками...")
    # Добавляем целевую переменную для анализа корреляций
    df_for_corr = df_clean.copy()
    if 'Sleep Disorder_encoded' in df_for_corr.columns:
        corr_matrix = analyzer.plot_correlation_matrix(
            df_for_corr,
            numerical_cols,
            'Sleep Disorder_encoded'
        )

    # 2.5 Зависимость ключевых признаков от целевой переменной
    print("\nАнализ зависимости признаков от целевой переменной...")
    # Удален блок с BMI Category, так как он уже закодирован

    # ШАГ 3: Обучение моделей машинного обучения
    print("\n" + "=" * 70)
    print("ШАГ 3: ОБУЧЕНИЕ МОДЕЛЕЙ МАШИННОГО ОБУЧЕНИЯ")
    print("=" * 70)

    trainer = ModelTrainer()

    # 3.1 Разделение данных
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)

    # 3.2 Инициализация и обучение моделей
    trainer.initialize_models()

    # Получаем названия классов из label_encoder (только Insomnia и Sleep Apnea)
    class_names = preprocessor.label_encoders['Sleep Disorder'].classes_

    results = trainer.train_models(
        X_train, X_test, y_train, y_test,
        use_grid_search=False,  # Временно отключаем GridSearch
        class_names=class_names
    )

    # 3.3 Визуализация результатов
    print("\nВизуализация результатов сравнения моделей...")
    metrics_df = trainer.plot_comparison()

    print("\nМатрицы ошибок для всех моделей...")
    trainer.plot_confusion_matrices(y_test, class_names)

    # 3.4 Анализ важности признаков
    print("\nАнализ важности признаков...")
    trainer.plot_feature_importance(X_train)

    # 3.5 Сохранение результатов
    trainer.save_results()

    # ШАГ 4: Выводы и рекомендации
    print("\n" + "=" * 70)
    print("ШАГ 4: ВЫВОДЫ И РЕКОМЕНДАЦИИ")
    print("=" * 70)

    if trainer.best_model:
        best_model_name = trainer.best_model['name']
        best_f1 = trainer.best_model['metrics']['f1_score']

        print(f"\nЛучшая модель: {best_model_name}")
        print(f"   F1-Score: {best_f1:.4f}")

        # Анализ важных признаков для лучшей модели
        if hasattr(trainer.results[best_model_name]['model'], 'feature_importances_'):
            print(f"\nКлючевые факторы, влияющие на расстройства сна:")

            model = trainer.results[best_model_name]['model']
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]

            feature_names = X_train.columns.tolist()

            print("\nТоп-10 важных признаков:")
            print("-" * 40)
            for i, idx in enumerate(indices[:10]):
                print(f"{i + 1:2d}. {feature_names[idx]:30s} : {importances[idx]:.4f}")

    print("\n" + "=" * 70)
    print("АНАЛИЗ УСПЕШНО ЗАВЕРШЕН!")
    print("=" * 70)

    # Возвращаем объекты для дальнейшего использования
    return {
        'preprocessor': preprocessor,
        'analyzer': analyzer,
        'trainer': trainer,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'results': results
    }


if __name__ == "__main__":
    # Запуск основного скрипта
    results = main()

    # Пример доступа к результатам
    print("\nИтоговые метрики моделей:")
    for model_name, metrics in results['trainer'].results.items():
        print(f"{model_name:20s}: F1 = {metrics['f1_score']:.3f}, Acc = {metrics['accuracy']:.3f}")