import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
import os
import warnings

warnings.filterwarnings('ignore')


class ModelTrainer:
    """Класс для обучения и оценки моделей машинного обучения"""

    def __init__(self, results_dir="../results"):
        self.results_dir = results_dir
        self.models_dir = os.path.join(results_dir, "models")
        self.reports_dir = os.path.join(results_dir, "reports")
        self.plots_dir = os.path.join(results_dir, "plots")

        # Создаем директории
        for dir_path in [self.models_dir, self.reports_dir, self.plots_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Инициализация моделей
        self.models = {}
        self.results = {}
        self.best_model = None
        self.scaler = StandardScaler()

    def split_data(self, X, y, test_size=0.25, random_state=42):
        """Разделение данных на обучающую и тестовую выборки"""
        print(f"\nРазделение данных (test_size={test_size})...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        print(f"Обучающая выборка: {X_train.shape}, {y_train.shape}")
        print(f"Тестовая выборка: {X_test.shape}, {y_test.shape}")

        # Масштабирование признаков
        print("Масштабирование признаков...")
        numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        X_test_scaled[numerical_cols] = self.scaler.transform(X_test[numerical_cols])

        return X_train_scaled, X_test_scaled, y_train, y_test

    def initialize_models(self):
        """Инициализация моделей для сравнения"""
        print("\nИнициализация моделей...")

        self.models = {
            "Logistic Regression": {
                "model": LogisticRegression(
                    max_iter=10000,  # Увеличиваем для сходимости
                    random_state=42,
                    multi_class='ovr',
                    class_weight='balanced'
                ),
                "params": {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'solver': ['lbfgs', 'liblinear']
                }
            },
            "Decision Tree": {
                "model": DecisionTreeClassifier(
                    random_state=42,
                    class_weight='balanced'
                ),
                "params": {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            "Random Forest": {
                "model": RandomForestClassifier(
                    random_state=42,
                    class_weight='balanced',
                    n_jobs=-1
                ),
                "params": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
            },
            "SVM": {
                "model": SVC(
                    random_state=42,
                    class_weight='balanced',
                    probability=True
                ),
                "params": {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.01, 0.1],
                    'kernel': ['rbf', 'linear']
                }
            }
        }

        print(f"Инициализировано {len(self.models)} моделей")
        return self.models

    def train_models(self, X_train, X_test, y_train, y_test, use_grid_search=True, class_names=None):
        """Обучение и оценка моделей"""
        print("\nОбучение моделей...")
        print("=" * 60)

        for model_name, model_info in self.models.items():
            print(f"\nМодель: {model_name}")
            print("-" * 40)

            if use_grid_search and model_info.get("params"):
                # Используем GridSearchCV для подбора гиперпараметров
                print("Подбор гиперпараметров с GridSearchCV...")

                grid_search = GridSearchCV(
                    estimator=model_info["model"],
                    param_grid=model_info["params"],
                    cv=5,
                    scoring='f1_weighted',
                    n_jobs=-1,
                    verbose=0
                )

                grid_search.fit(X_train, y_train)

                # Сохраняем лучшую модель
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_score = grid_search.best_score_

                print(f"Лучшие параметры: {best_params}")
                print(f"Лучший CV score: {best_score:.4f}")

            else:
                # Обычное обучение без подбора параметров
                print("⚡ Обучение без подбора параметров...")
                best_model = model_info["model"]
                best_model.fit(X_train, y_train)
                best_params = "default"
                best_score = cross_val_score(
                    best_model, X_train, y_train,
                    cv=5, scoring='f1_weighted'
                ).mean()

                print(f"CV score: {best_score:.4f}")

            # Предсказания на тестовой выборке
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test) if hasattr(best_model, "predict_proba") else None

            # Расчет метрик
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            # ROC-AUC для многоклассовой классификации
            roc_auc = None
            if y_pred_proba is not None and len(np.unique(y_test)) > 1:
                try:
                    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                except:
                    roc_auc = None

            # Сохранение результатов
            self.results[model_name] = {
                "model": best_model,
                "predictions": y_pred,
                "probabilities": y_pred_proba,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc,
                "best_params": best_params,
                "cv_score": best_score
            }

            # Вывод отчета по классификации
            print(f"\nРезультаты на тестовой выборке:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            if roc_auc:
                print(f"ROC-AUC: {roc_auc:.4f}")

            # Детальный отчет по классам
            print("\nДетальный отчет по классам:")
            if class_names is None:
                # Если названия классов не переданы, используем числовые обозначения
                class_names = [str(i) for i in range(len(np.unique(y_train)))]
            print(classification_report(y_test, y_pred,
                                        target_names=class_names))

        # Определение лучшей модели по F1-Score
        self._select_best_model()

        return self.results

    def _select_best_model(self):
        """Выбор лучшей модели на основе F1-Score"""
        if self.results:
            best_model_name = max(self.results.items(),
                                  key=lambda x: x[1]['f1_score'])[0]
            self.best_model = {
                "name": best_model_name,
                "model": self.results[best_model_name]["model"],
                "metrics": {
                    "accuracy": self.results[best_model_name]["accuracy"],
                    "f1_score": self.results[best_model_name]["f1_score"]
                }
            }

            print(f"\nЛУЧШАЯ МОДЕЛЬ: {best_model_name}")
            print(f"   F1-Score: {self.results[best_model_name]['f1_score']:.4f}")
            print(f"   Accuracy: {self.results[best_model_name]['accuracy']:.4f}")

    def plot_comparison(self):
        """Визуализация сравнения моделей"""
        if not self.results:
            print("Нет результатов для визуализации!")
            return

        # Подготовка данных для визуализации
        metrics_data = []
        for model_name, metrics in self.results.items():
            metrics_data.append({
                "Model": model_name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-Score": metrics["f1_score"]
            })

        df_metrics = pd.DataFrame(metrics_data)

        # 1. Барчарт сравнения метрик
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
        colors = plt.cm.Set2(np.linspace(0, 1, len(df_metrics)))

        for idx, metric in enumerate(metric_names):
            ax = axes[idx]
            bars = ax.bar(df_metrics["Model"], df_metrics[metric], color=colors)

            # Добавляем значения на бары
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=10)

            ax.set_title(f'{metric} по моделям', fontsize=14)
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.suptitle('Сравнение моделей машинного обучения', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "models_comparison.png"),
                    dpi=300, bbox_inches='tight')
        plt.show()

        # 2. Heatmap метрик
        plt.figure(figsize=(12, 6))
        metrics_df_plot = df_metrics.set_index("Model").T

        sns.heatmap(metrics_df_plot, annot=True, fmt='.3f', cmap='YlOrRd',
                    linewidths=1, linecolor='black', cbar_kws={'label': 'Значение'})

        plt.title('Тепловая карта метрик качества моделей', fontsize=16, pad=20)
        plt.xlabel('Модель')
        plt.ylabel('Метрика')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "metrics_heatmap.png"),
                    dpi=300, bbox_inches='tight')
        plt.show()

        return df_metrics

    def plot_confusion_matrices(self, y_test, class_names=None):
        """Визуализация матриц ошибок для всех моделей"""
        if not self.results:
            print("Нет результатов для визуализации!")
            return

        n_models = len(self.results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        if class_names is None:
            # Если названия классов не переданы, создаем их
            unique_classes = sorted(np.unique(y_test))
            class_names = [f'Class {c}' for c in unique_classes]

        for idx, (model_name, metrics) in enumerate(self.results.items()):
            if idx < len(axes):
                ax = axes[idx]

                cm = confusion_matrix(y_test, metrics["predictions"])

                # Нормализованная матрица ошибок
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

                # Heatmap
                im = ax.imshow(cm_normalized, interpolation='nearest',
                               cmap=plt.cm.Blues, vmin=0, vmax=1)
                ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # Добавляем текст в ячейки
                thresh = cm_normalized.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j]:.2f})",
                                ha="center", va="center",
                                color="white" if cm_normalized[i, j] > thresh else "black",
                                fontsize=10)

                ax.set_title(f'{model_name}\nAccuracy: {metrics["accuracy"]:.3f}',
                             fontsize=14, pad=20)
                ax.set_xlabel('Предсказанный класс')
                ax.set_ylabel('Истинный класс')
                ax.set_xticks(np.arange(len(class_names)))
                ax.set_yticks(np.arange(len(class_names)))
                ax.set_xticklabels(class_names)
                ax.set_yticklabels(class_names)

                # Поворачиваем подписи осей X
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Скрываем лишние subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Матрицы ошибок для всех моделей', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "confusion_matrices.png"),
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_feature_importance(self, X_train, top_n=15):
        """Визуализация важности признаков для tree-based моделей"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, (model_name, metrics) in enumerate(self.results.items()):
            if idx < len(axes):
                model = metrics["model"]

                # Проверяем, есть ли у модели атрибут feature_importances_
                if hasattr(model, 'feature_importances_'):
                    ax = axes[idx]

                    # Получаем важность признаков
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1][:top_n]

                    # Создаем DataFrame для визуализации
                    feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else \
                        [f'Feature {i}' for i in range(X_train.shape[1])]

                    importance_df = pd.DataFrame({
                        'Feature': [feature_names[i] for i in indices],
                        'Importance': importances[indices]
                    })

                    # Горизонтальный барчарт
                    bars = ax.barh(range(len(indices)), importance_df['Importance'][:top_n])
                    ax.set_yticks(range(len(indices)))
                    ax.set_yticklabels(importance_df['Feature'][:top_n])
                    ax.invert_yaxis()  # Самый важный признак вверху

                    # Добавляем значения на бары
                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        ax.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                                f'{width:.3f}', ha='left', va='center', fontsize=9)

                    ax.set_xlabel('Важность признака')
                    ax.set_title(f'{model_name}\nТоп-{top_n} важных признаков',
                                 fontsize=14, pad=20)
                    ax.grid(axis='x', alpha=0.3)

                    # Сохраняем важность признаков в файл
                    importance_df.to_csv(
                        os.path.join(self.reports_dir, f"feature_importance_{model_name}.csv"),
                        index=False, encoding='utf-8-sig'
                    )
                else:
                    axes[idx].set_visible(False)
                    print(f"⚠️  Модель {model_name} не поддерживает feature_importances_")

        plt.suptitle('Важность признаков в tree-based моделях', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "feature_importance.png"),
                    dpi=300, bbox_inches='tight')
        plt.show()

    def save_results(self):
        """Сохранение результатов обучения"""
        print(f"\nСохранение результатов...")

        # 1. Сохранение метрик моделей
        metrics_data = []
        for model_name, metrics in self.results.items():
            metrics_data.append({
                "Model": model_name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1_Score": metrics["f1_score"],
                "ROC_AUC": metrics.get("roc_auc", "N/A"),
                "CV_Score": metrics.get("cv_score", "N/A"),
                "Best_Params": str(metrics.get("best_params", "N/A"))
            })

        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(
            os.path.join(self.reports_dir, "models_metrics.csv"),
            index=False, encoding='utf-8-sig'
        )
        print(f"Метрики сохранены: models_metrics.csv")

        # 2. Сохранение лучшей модели
        if self.best_model:
            model_path = os.path.join(self.models_dir, "best_model.pkl")
            joblib.dump(self.best_model["model"], model_path)
            print(f"Лучшая модель сохранена: {model_path}")

            # Сохраняем scaler
            scaler_path = os.path.join(self.models_dir, "scaler.pkl")
            joblib.dump(self.scaler, scaler_path)
            print(f"Scaler сохранен: {scaler_path}")

        # 3. Сохранение сводного отчета
        report_path = os.path.join(self.reports_dir, "training_summary.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("ОТЧЕТ ОБ ОБУЧЕНИИ МОДЕЛЕЙ\n")
            f.write("=" * 60 + "\n\n")

            f.write("ЛУЧШАЯ МОДЕЛЬ:\n")
            f.write("-" * 40 + "\n")
            if self.best_model:
                f.write(f"Название: {self.best_model['name']}\n")
                f.write(f"Accuracy: {self.best_model['metrics']['accuracy']:.4f}\n")
                f.write(f"F1-Score: {self.best_model['metrics']['f1_score']:.4f}\n\n")

            f.write("ВСЕ МОДЕЛИ:\n")
            f.write("-" * 40 + "\n")
            for model_name, metrics in self.results.items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall:    {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score:  {metrics['f1_score']:.4f}\n")
                if metrics.get('roc_auc'):
                    f.write(f"  ROC-AUC:   {metrics['roc_auc']:.4f}\n")

        print(f"Сводный отчет сохранен: {report_path}")
        print(f"Все результаты сохранены в папке: {self.results_dir}")