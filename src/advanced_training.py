import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings('ignore')


class AdvancedModelTrainer:
    """Расширенный тренер моделей с дополнительными возможностями"""

    def __init__(self, results_dir="../results"):
        self.results_dir = results_dir

    def get_extended_models(self):
        """Получение расширенного набора моделей"""
        models = {
            "Logistic Regression": {
                "model": LogisticRegression(max_iter=10000, random_state=42, class_weight='balanced'),
                "params": {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}
            },
            "Random Forest": {
                "model": RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
                "params": {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}
            },
            "XGBoost": {
                "model": xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
                "params": {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
            },
            "LightGBM": {
                "model": lgb.LGBMClassifier(random_state=42, verbose=-1),
                "params": {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
            },
            "SVM": {
                "model": SVC(random_state=42, class_weight='balanced', probability=True),
                "params": {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
            }
        }
        return models

    def compare_balancing_strategies(self, X_train, y_train, X_test, y_test):
        """Сравнение разных методов балансировки классов"""
        print("🔧 Сравнение методов балансировки классов...")

        strategies = {
            'Без балансировки': None,
            'SMOTE': SMOTE(random_state=42),
            'UnderSampling': RandomUnderSampler(random_state=42),
            'SMOTE + UnderSampling': Pipeline([
                ('smote', SMOTE(random_state=42)),
                ('under', RandomUnderSampler(random_state=42))
            ])
        }

        results = []
        for strategy_name, sampler in strategies.items():
            print(f"\n📊 Стратегия: {strategy_name}")

            if sampler:
                X_balanced, y_balanced = sampler.fit_resample(X_train, y_train)
                print(f"   Размер после балансировки: {len(X_balanced)}")
                print(f"   Распределение классов: {np.bincount(y_balanced)}")
            else:
                X_balanced, y_balanced = X_train, y_train

            # Обучаем Random Forest для сравнения
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(random_state=42, n_estimators=100)
            model.fit(X_balanced, y_balanced)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            results.append({
                'Стратегия': strategy_name,
                'Accuracy': accuracy,
                'F1-Score': f1,
                'Размер выборки': len(X_balanced)
            })

        # Визуализация результатов
        results_df = pd.DataFrame(results)

        plt.figure(figsize=(10, 6))
        x = range(len(results_df))
        width = 0.35

        plt.bar(x, results_df['Accuracy'], width, label='Accuracy', color='skyblue')
        plt.bar([i + width for i in x], results_df['F1-Score'], width, label='F1-Score', color='lightgreen')

        plt.xlabel('Стратегия балансировки')
        plt.ylabel('Метрика')
        plt.title('Сравнение методов балансировки классов')
        plt.xticks([i + width / 2 for i in x], results_df['Стратегия'], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/plots/balancing_strategies.png", dpi=300)
        plt.show()

        return results_df

    def plot_roc_pr_curves(self, model, X, y, model_name):
        """Построение ROC и Precision-Recall кривых"""
        print(f"📈 Построение ROC/PR кривых для {model_name}...")

        # Кросс-валидация для получения вероятностей
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')

        # ROC кривые
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Для каждого класса
        for i in range(y_pred_proba.shape[1]):
            fpr, tpr, _ = roc_curve((y == i).astype(int), y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax1.plot(fpr, tpr, lw=2, label=f'Класс {i} (AUC = {roc_auc:.3f})')

        ax1.plot([0, 1], [0, 1], 'k--', lw=2)
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'ROC кривые - {model_name}')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)

        # Precision-Recall кривые
        for i in range(y_pred_proba.shape[1]):
            precision, recall, _ = precision_recall_curve((y == i).astype(int), y_pred_proba[:, i])
            pr_auc = auc(recall, precision)
            ax2.plot(recall, precision, lw=2, label=f'Класс {i} (AUC = {pr_auc:.3f})')

        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(f'Precision-Recall кривые - {model_name}')
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/plots/roc_pr_{model_name}.png", dpi=300)
        plt.show()