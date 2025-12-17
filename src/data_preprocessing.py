import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Класс для предобработки данных о здоровье сна"""

    def __init__(self, data_path):
        """
        Инициализация предобработчика

        Parameters:
        -----------
        data_path : str
            Путь к CSV файлу с данными
        """
        self.data_path = data_path
        self.df = None
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def load_data(self):
        """Загрузка данных из CSV файла"""
        print("Загрузка данных...")
        self.df = pd.read_csv(self.data_path)
        print(f"Данные загружены. Размер: {self.df.shape}")
        print(f"Столбцы: {list(self.df.columns)}")
        return self.df

    def explore_data(self):
        """Предварительный анализ данных"""
        print("\nАнализ структуры данных:")
        print("=" * 50)

        # Основная информация
        print("Информация о типах данных:")
        self.df.info()

        print("\nСтатистика числовых признаков:")
        print(self.df.describe())

        print("\nАнализ пропущенных значений:")
        missing_data = self.df.isnull().sum()
        percent_missing = (missing_data / len(self.df)) * 100
        missing_table = pd.concat([missing_data, percent_missing],
                                  axis=1, keys=['Количество', 'Процент'])
        print(missing_table[missing_table['Количество'] > 0])

        print(f"\nРаспределение целевой переменной (Sleep Disorder):")
        print(self.df['Sleep Disorder'].value_counts())
        print(self.df['Sleep Disorder'].value_counts(normalize=True) * 100)

        return missing_table

    def clean_data(self):
        """Очистка и преобразование данных"""
        print("\nОчистка данных...")

        # 1. Удаление дубликатов
        initial_size = len(self.df)
        self.df = self.df.drop_duplicates()
        duplicates_removed = initial_size - len(self.df)
        print(f"Удалено дубликатов: {duplicates_removed}")

        # 2. УДАЛЕНИЕ СТРОК С ПРОПУСКАМИ В ЦЕЛЕВОЙ ПЕРЕМЕННОЙ
        print(f"Удаление строк с пропусками в Sleep Disorder...")
        initial_rows = len(self.df)
        self.df = self.df.dropna(subset=['Sleep Disorder'])
        rows_removed = initial_rows - len(self.df)
        print(f"Удалено строк с пропусками: {rows_removed}")
        print(f"Осталось записей: {len(self.df)}")

        # 3. Преобразование давления в два числовых признака
        print("Преобразование давления...")
        self.df[['Systolic_BP', 'Diastolic_BP']] = (
            self.df['Blood Pressure']
            .str.split('/', expand=True)
            .astype(int)
        )
        self.df = self.df.drop('Blood Pressure', axis=1)

        # 3. Кодирование категориальных переменных
        print("Кодирование категориальных признаков...")

        # Для пола
        self.label_encoders['Gender'] = LabelEncoder()
        self.df['Gender_encoded'] = self.label_encoders['Gender'].fit_transform(
            self.df['Gender']
        )

        # Для целевой переменной
        self.label_encoders['Sleep Disorder'] = LabelEncoder()
        self.df['Sleep Disorder_encoded'] = self.label_encoders['Sleep Disorder'].fit_transform(
            self.df['Sleep Disorder']
        )

        # Для BMI Category - порядковое кодирование
        print("Кодирование BMI Category...")
        self.label_encoders['BMI Category'] = LabelEncoder()
        self.df['BMI_Category_encoded'] = self.label_encoders['BMI Category'].fit_transform(
            self.df['BMI Category']
        )

        # One-Hot Encoding для профессии (более 10 уникальных значений)
        print("One-Hot Encoding для профессии...")
        occupation_dummies = pd.get_dummies(
            self.df['Occupation'],
            prefix='Occ',
            drop_first=False
        )
        self.df = pd.concat([self.df, occupation_dummies], axis=1)

        # 4. Удаление ненужных столбцов
        columns_to_drop = ['Person ID', 'Gender', 'Sleep Disorder', 'Occupation', 'BMI Category']
        self.df = self.df.drop(
            [col for col in columns_to_drop if col in self.df.columns],
            axis=1
        )

        print(f"Очистка завершена. Новый размер: {self.df.shape}")
        print(f"Новые столбцы: {list(self.df.columns)}")

        return self.df

    def prepare_features_target(self):
        """Подготовка признаков и целевой переменной"""
        print("\nПодготовка признаков и целевой переменной...")

        # Целевая переменная - закодированные расстройства сна
        target_col = 'Sleep Disorder_encoded'
        if target_col in self.df.columns:
            y = self.df[target_col]
            X = self.df.drop(target_col, axis=1)

            # Удаляем исходные столбцы, если они есть
            if 'Sleep Disorder' in X.columns:
                X = X.drop('Sleep Disorder', axis=1)
            if 'Sleep Disorder_encoded' in X.columns:
                X = X.drop('Sleep Disorder_encoded', axis=1)

            print(f"Размер X: {X.shape}")
            print(f"Размер y: {y.shape}")
            print(f"Распределение классов в y: \n{y.value_counts()}")

            return X, y
        else:
            raise ValueError(f"Столбец {target_col} не найден в данных")

    def get_numerical_categorical_cols(self, X):
        """Получение списков числовых и категориальных столбцов"""
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        print(f"\nЧисловые признаки ({len(numerical_cols)}): {numerical_cols}")
        print(f"Категориальные признаки ({len(categorical_cols)}): {categorical_cols}")

        return numerical_cols, categorical_cols

    def save_processed_data(self, output_path):
        """Сохранение обработанных данных"""
        self.df.to_csv(output_path, index=False)
        print(f"Обработанные данные сохранены в {output_path}")


if __name__ == "__main__":
    # Тестирование модуля
    preprocessor = DataPreprocessor("data/Sleep_health_and_lifestyle_dataset.csv")
    df = preprocessor.load_data()
    preprocessor.explore_data()
    df_clean = preprocessor.clean_data()
    X, y = preprocessor.prepare_features_target()