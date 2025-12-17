import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from scipy import stats


class EDAAnalyzer:
    """Класс для визуализации и разведочного анализа данных"""

    def __init__(self, output_dir="../results/plots"):
        self.output_dir = output_dir
        self.setup_visualization()

        # Создаем директорию для сохранения графиков
        os.makedirs(output_dir, exist_ok=True)

    def setup_visualization(self):
        """Настройка параметров визуализации"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("Set2")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 14

    def save_plot(self, filename):
        """Сохранение графика в файл"""
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"График сохранен: {filepath}")

    def plot_target_distribution(self, y, label_encoder=None):
        """Визуализация распределения целевой переменной"""
        plt.figure(figsize=(10, 6))

        # Если есть кодировщик, используем оригинальные названия
        if label_encoder:
            labels = label_encoder.classes_
            counts = y.value_counts().sort_index()
            bars = plt.bar(range(len(labels)), counts.values, color=sns.color_palette("Set2"))

            # Добавляем процент на каждый бар
            total = len(y)
            for i, (bar, count) in enumerate(zip(bars, counts.values)):
                height = bar.get_height()
                percentage = (count / total) * 100
                plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                         f'{count}\n({percentage:.1f}%)',
                         ha='center', va='bottom', fontsize=11)

            plt.xticks(range(len(labels)), labels, rotation=0)
        else:
            counts = y.value_counts()
            bars = plt.bar(counts.index.astype(str), counts.values, color=sns.color_palette("Set2"))

            total = len(y)
            for bar, count in zip(bars, counts.values):
                height = bar.get_height()
                percentage = (count / total) * 100
                plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                         f'{count}\n({percentage:.1f}%)',
                         ha='center', va='bottom', fontsize=11)

        plt.title('Распределение категорий расстройств сна', fontsize=16, pad=20)
        plt.xlabel('Категория расстройства')
        plt.ylabel('Количество наблюдений')
        plt.grid(axis='y', alpha=0.3)

        self.save_plot("target_distribution.png")
        plt.show()

    def plot_numerical_distributions(self, df, numerical_cols, n_cols=3):
        """Визуализация распределения числовых признаков"""
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        axes = axes.flatten()

        for idx, col in enumerate(numerical_cols):
            if idx < len(axes):
                ax = axes[idx]

                # Гистограмма с KDE
                sns.histplot(df[col], kde=True, ax=ax, color='skyblue',
                             edgecolor='black', alpha=0.7)

                # Выбросы (за пределами 1.5 * IQR)
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]

                # Статистики
                mean_val = df[col].mean()
                median_val = df[col].median()

                ax.axvline(mean_val, color='red', linestyle='--',
                           linewidth=2, label=f'Среднее: {mean_val:.2f}')
                ax.axvline(median_val, color='green', linestyle='--',
                           linewidth=2, label=f'Медиана: {median_val:.2f}')

                ax.set_title(f'{col}\n(n={len(df)}, выбросов: {len(outliers)})')
                ax.set_xlabel('')
                ax.set_ylabel('Частота')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

        # Скрываем лишние subplots
        for idx in range(len(numerical_cols), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Распределение числовых признаков с выделением выбросов',
                     fontsize=16, y=1.02)
        plt.tight_layout()

        self.save_plot("numerical_distributions.png")
        plt.show()

    def plot_boxplots_outliers(self, df, numerical_cols, n_cols=3):
        """Диаграммы размаха для анализа выбросов"""
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        axes = axes.flatten()

        for idx, col in enumerate(numerical_cols):
            if idx < len(axes):
                ax = axes[idx]

                # Boxplot
                bp = ax.boxplot(df[col].dropna(), vert=True, patch_artist=True,
                                boxprops=dict(facecolor='lightblue', color='darkblue'),
                                medianprops=dict(color='red', linewidth=2),
                                whiskerprops=dict(color='darkblue'),
                                capprops=dict(color='darkblue'),
                                flierprops=dict(marker='o', color='red',
                                                alpha=0.5, markersize=5))

                # Добавляем количество выбросов
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers_count = len(df[(df[col] < Q1 - 1.5 * IQR) |
                                        (df[col] > Q3 + 1.5 * IQR)])

                ax.set_title(f'{col}\nВыбросов: {outliers_count}', fontsize=12)
                ax.set_ylabel('Значение')
                ax.grid(True, alpha=0.3)

                # Добавляем аннотацию с границами выбросов
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                ax.text(0.02, 0.98, f'Q1: {Q1:.1f}\nQ3: {Q3:.1f}\nIQR: {IQR:.1f}',
                        transform=ax.transAxes, fontsize=9,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Скрываем лишние subplots
        for idx in range(len(numerical_cols), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Анализ выбросов в числовых признаках (Boxplot)',
                     fontsize=16, y=1.02)
        plt.tight_layout()

        self.save_plot("boxplots_outliers.png")
        plt.show()

    def plot_correlation_matrix(self, df, numerical_cols, target_col=None):
        """Тепловая карта корреляций"""
        # Выбираем только числовые столбцы
        numeric_df = df[numerical_cols].copy()

        if target_col and target_col in df.columns:
            numeric_df[target_col] = df[target_col]

        # Вычисляем корреляционную матрицу
        corr_matrix = numeric_df.corr()

        plt.figure(figsize=(14, 10))

        # Маска для скрытия верхнего треугольника
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Тепловая карта
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                    cmap='coolwarm', center=0, square=True,
                    linewidths=0.5, cbar_kws={"shrink": 0.8},
                    annot_kws={"size": 9})

        plt.title('Тепловая карта корреляций между признаками',
                  fontsize=16, pad=20)

        # Поворачиваем подписи осей
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        self.save_plot("correlation_matrix.png")
        plt.show()

        # Анализ корреляций с целевой переменной
        if target_col:
            print(f"\nКорреляции с целевой переменной ({target_col}):")
            target_correlations = corr_matrix[target_col].sort_values(
                key=abs, ascending=False
            )
            print(target_correlations)

            # Визуализация топ-10 корреляций
            top_n = min(10, len(target_correlations) - 1)
            top_features = target_correlations.drop(target_col).head(top_n)

            plt.figure(figsize=(12, 6))
            colors = ['red' if x < 0 else 'green' for x in top_features.values]
            bars = plt.barh(top_features.index, top_features.values, color=colors)

            # Добавляем значения на бары
            for bar, val in zip(bars, top_features.values):
                width = bar.get_width()
                plt.text(width if width > 0 else width - 0.01,
                         bar.get_y() + bar.get_height() / 2,
                         f'{val:.3f}', ha='left' if width > 0 else 'right',
                         va='center', fontsize=10, color='black',
                         fontweight='bold')

            plt.axvline(x=0, color='black', linewidth=0.8)
            plt.title(f'Топ-{top_n} признаков по корреляции с {target_col}',
                      fontsize=16, pad=20)
            plt.xlabel('Коэффициент корреляции')
            plt.ylabel('Признак')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()

            self.save_plot("top_correlations.png")
            plt.show()

        return corr_matrix

    def plot_pairplot_selected(self, df, columns, target_col, sample_size=200):
        """Парные диаграммы рассеяния для выбранных признаков"""
        if len(df) > sample_size:
            df_sample = df.sample(sample_size, random_state=42)
        else:
            df_sample = df.copy()

        # Создаем pairplot только с выбранными столбцами
        plot_cols = columns + [target_col]
        plot_data = df_sample[plot_cols].copy()

        # Для лучшей визуализации, если много классов, используем hue
        if plot_data[target_col].nunique() <= 5:
            plt.figure(figsize=(14, 10))
            g = sns.pairplot(plot_data, hue=target_col,
                             palette='Set2', plot_kws={'alpha': 0.6})
            g.fig.suptitle(f'Парные диаграммы рассеяния (hue: {target_col})',
                           y=1.02, fontsize=16)
        else:
            plt.figure(figsize=(14, 10))
            g = sns.pairplot(plot_data, palette='Set2',
                             plot_kws={'alpha': 0.6})
            g.fig.suptitle('Парные диаграммы рассеяния',
                           y=1.02, fontsize=16)

        self.save_plot("pairplot_selected.png")
        plt.show()

    def plot_feature_vs_target(self, df, feature_col, target_col):
        """Визуализация зависимости признака от целевой переменной"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 1. Boxplot
        sns.boxplot(x=target_col, y=feature_col, data=df,
                    ax=ax1, palette='Set2')
        ax1.set_title(f'Распределение {feature_col} по классам {target_col}')
        ax1.set_xlabel(target_col)
        ax1.set_ylabel(feature_col)
        ax1.grid(True, alpha=0.3)

        # 2. Violin plot
        sns.violinplot(x=target_col, y=feature_col, data=df,
                       ax=ax2, palette='Set2', inner='quartile')
        ax2.set_title(f'Плотность распределения {feature_col} по классам')
        ax2.set_xlabel(target_col)
        ax2.set_ylabel(feature_col)
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f'Анализ зависимости: {feature_col} → {target_col}',
                     fontsize=16)
        plt.tight_layout()

        self.save_plot(f"feature_vs_target_{feature_col}.png")
        plt.show()


if __name__ == "__main__":
    # Тестирование модуля
    analyzer = EDAAnalyzer()

    # Пример использования:
    # Создаем тестовые данные
    test_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.exponential(1, 100),
        'target': np.random.choice([0, 1, 2], 100)
    })

    analyzer.plot_target_distribution(test_data['target'])
    analyzer.plot_numerical_distributions(test_data, ['feature1', 'feature2'])