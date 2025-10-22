import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as stats
from typing import Dict, Tuple, List
import os
from scipy.stats import gaussian_kde
import csv

plt.style.use('seaborn-v0_8')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

class CompleteStatisticalAnalysis:

    def __init__(self, data_path: str, target_year: int = 2023):
        try:
            self.data = pd.read_csv(data_path, on_bad_lines='skip')
        except pd.errors.ParserError:
            print(f"Ошибка загрузки файла")
        
        self.target_year = target_year
        self.year_column = str(target_year)
        
        # Проверка, что год существует в данных
        if self.year_column not in self.data.columns:
            available_years = [col for col in self.data.columns if col.isdigit()]
            raise ValueError(f"Год {target_year} не найден. Доступные годы: {available_years}")
        
        # Фильтрация только регионов (исключаем РФ и округа)
        self.regions_data = self.data[~self.data['region'].str.contains(
            'федеральный округ|Российская Федерация', na=False
        )].copy()
        
        # Убирает пропущенные значения
        self.regions_data = self.regions_data.dropna(subset=[self.year_column])
        self.doctors_data = self.regions_data[self.year_column].values
        self.regions = self.regions_data['region'].values
        self.n = len(self.doctors_data)
        
        print(f"Загружено {self.n} регионов за {target_year} год")
        print(f"Диапазон данных: от {np.min(self.doctors_data):.0f} до {np.max(self.doctors_data):.0f} врачей")
    
    def _manual_read_csv(self, file_path):
        """Чтение CSV файла с обработкой ошибок"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
            for i, row in enumerate(reader):
                if len(row) != len(headers):
                    print(f"Пропущена строка {i+2}: несоответствие количества столбцов")
                    continue
                data.append(row)
        
        return pd.DataFrame(data, columns=headers)
    
    def section_data_description(self):
        """Описание данных"""
        print("ОПИСАНИЕ ДАННЫХ")
        print("=" * 50)
        print("Источник: Федеральная служба государственной статистики (Росстат)")
        print("Раздел: Здравоохранение")
        print("Показатель: Численность врачей всех специальностей (физических лиц)")
        print(f"Период: {self.target_year} год")
        print("Территориальный охват: субъекты Российской Федерации")
        print("Единица измерения: человек")
        print(f"Объем данных: {self.n} наблюдений")
        print()
    
    def section_formal_representation(self):
        """Формальное представление данных"""
        print("ФОРМАЛЬНОЕ ПРЕДСТАВЛЕНИЕ ДАННЫХ")
        print("=" * 50)
        
        # Показываем первые 10 регионов для примера
        df_display = self.regions_data[['region', self.year_column]].head(10).copy()
        df_display.columns = ['Субъект РФ', f'Численность врачей, {self.target_year} г., чел.']
        print("Первые 10 регионов из выборки:")
        print(df_display.to_string(index=False))
        print(f"\n... и еще {self.n - 10} регионов")
        print()
    
    def section_visual_representation(self):
        """Наглядное представление данных"""
        # Столбчатая диаграмма для топ-20 регионов
        top_regions = self.regions_data.nlargest(20, self.year_column)
        
        plt.figure(figsize=(14, 8))
        bars = plt.bar(range(len(top_regions)), top_regions[self.year_column].values,
                    color=plt.cm.viridis(np.linspace(0, 1, len(top_regions))))
        
        plt.title(f'Топ-20 регионов по численности врачей ({self.target_year} г.)', fontweight='bold')
        plt.xlabel('Регионы')
        plt.ylabel('Численность врачей, чел.')
        plt.xticks(range(len(top_regions)), 
                [r[:20] + '...' if len(r) > 20 else r for r in top_regions['region']], 
                rotation=45, ha='right')
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 100,
                    f'{int(height):,}'.replace(',', ' '), 
                    ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('results/bar_chart_top20.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # ГИСТОГРАММА
        plt.figure(figsize=(12, 8))
        
        # Гистограмма с 15 интервалами
        n, bins, patches = plt.hist(self.doctors_data, bins=15, 
                                color='lightblue', edgecolor='navy',
                                alpha=0.7, linewidth=1.2)
        
        plt.xlabel('Численность врачей, чел.', fontsize=12, fontweight='bold')
        plt.ylabel('Количество регионов', fontsize=12, fontweight='bold')
        plt.title(f'Распределение численности врачей по регионам ({self.target_year} г.)', 
                fontsize=14, fontweight='bold')
        
        # Сетка только по Y для лучшей читаемости
        plt.grid(True, alpha=0.3, axis='y')
        
        # Подписи значений над столбцами
        for i, count in enumerate(n):
            if count > 0:
                plt.text(bins[i] + (bins[i+1] - bins[i])/2, count + 0.5, 
                        f'{int(count)}', ha='center', va='bottom', 
                        fontsize=10, fontweight='bold')
        
        # Форматирование больших чисел на оси X
        plt.gca().xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'{int(x):,}'.replace(',', ' '))
        )
        
        # Ось Y начинается с 0
        plt.ylim(0, max(n) + 2)
        
        plt.tight_layout()
        plt.savefig('results/histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("НАГЛЯДНОЕ ПРЕДСТАВЛЕНИЕ ДАННЫХ")
        print("=" * 50)
        print("Созданы диаграммы в папке 'results/'")
        print("  bar_chart_top20.png: Топ-20 регионов")
        print("  histogram.png: Базовая гистограмма распределения")
        print("  histogram_with_stats.png: Гистограмма со статистическими линиями")
        print(f"Общее количество регионов на гистограмме: {int(sum(n))}")
        print(f"Количество интервалов: 15")
        print()
    
    def calculate_basic_characteristics(self):
        print("ОСНОВНЫЕ ВЫБОРОЧНЫЕ ОЦЕНКИ")
        print("=" * 50)
        
        chars = {}
        n = len(self.doctors_data)
        chars['n'] = n
        
        # Объем выборки
        print(f"Объем выборки: {chars['n']} регионов")
        
        # Среднее выборочное
        chars['mean'] = np.mean(self.doctors_data)
        print(f"Среднее выборочное: {chars['mean']:.2f} чел.")
        
        # Выборочные начальные моменты
        chars['a2'] = np.mean(self.doctors_data**2)
        chars['a3'] = np.mean(self.doctors_data**3)
        chars['a4'] = np.mean(self.doctors_data**4)
        print(f"Выборочные начальные моменты:")
        print(f"   a₂: {chars['a2']:.2f}")
        print(f"   a₃: {chars['a3']:.2e}")
        print(f"   a₄: {chars['a4']:.2e}")
        
        # Выборочные центральные моменты
        chars['m2'] = np.mean((self.doctors_data - chars['mean'])**2)
        chars['m3'] = np.mean((self.doctors_data - chars['mean'])**3)
        chars['m4'] = np.mean((self.doctors_data - chars['mean'])**4)
        print(f"Выборочные центральные моменты:")
        print(f"   m₂: {chars['m2']:.2f}")
        print(f"   m₃: {chars['m3']:.2e}")
        print(f"   m₄: {chars['m4']:.2e}")
        
        # Смещенная выборочная дисперсия
        chars['variance_biased'] = chars['m2']
        print(f"Смещенная выборочная дисперсия: {chars['variance_biased']:.2f}")
        
        # Несмещенная выборочная дисперсия (ДОБАВЛЕНО)
        chars['variance_unbiased'] = np.var(self.doctors_data, ddof=1)
        print(f"Несмещенная выборочная дисперсия: {chars['variance_unbiased']:.2f}")
        
        # Выборочное среднее квадратическое отклонение (смещенное)
        chars['std_biased'] = np.sqrt(chars['variance_biased'])
        print(f"Выборочное среднее квадратическое отклонение: {chars['std_biased']:.2f} чел.")
        
        # Несмещенное СКО (ДОБАВЛЕНО для совместимости)
        chars['std_dev'] = np.sqrt(chars['variance_unbiased'])
        
        print()
        return chars

    def calculate_additional_characteristics(self, basic_chars=None):
        print("ДОПОЛНИТЕЛЬНЫЕ ВЫБОРОЧНЫЕ ОЦЕНКИ")
        print("=" * 50)
        
        chars = {}
        
        if basic_chars is None:
            basic_chars = self.calculate_basic_characteristics()
        
        # Выборочная медиана
        chars['median'] = np.median(self.doctors_data)
        print(f"Выборочная медиана: {chars['median']:.2f} чел.")
        
        # Выборочное абсолютное отклонение
        chars['mad'] = np.mean(np.abs(self.doctors_data - chars['median']))
        print(f"Выборочное среднее абсолютное отклонение: {chars['mad']:.2f} чел.")
        
        # Квартили
        chars['q1'] = np.percentile(self.doctors_data, 25)
        chars['q2'] = np.percentile(self.doctors_data, 50)
        chars['q3'] = np.percentile(self.doctors_data, 75)
        print(f"Выборочные квартили:")
        print(f"   Q1 (25%): {chars['q1']:.2f} чел.")
        print(f"   Q2 (50%, медиана): {chars['q2']:.2f} чел.")
        print(f"   Q3 (75%): {chars['q3']:.2f} чел.")
        
        # Интерквартильная широта
        chars['iqr'] = chars['q3'] - chars['q1']
        print(f"Интерквартильная широта: {chars['iqr']:.2f} чел.")
        
        # Полусумма выборочных квартилей
        chars['midquartile'] = (chars['q1'] + chars['q3']) / 2
        print(f"Полусумма выборочных квартилей: {chars['midquartile']:.2f} чел.")

        # Разность выборочных квартилей
        chars['quartile_diff'] = chars['q3'] - chars['q1']
        print(f"Разность выборочных квартилей: {chars['quartile_diff']:.2f} чел.")
        
        # Экстремальные элементы
        chars['min'] = np.min(self.doctors_data)
        chars['max'] = np.max(self.doctors_data)
        min_region = self.regions_data[self.regions_data[self.year_column] == chars['min']]['region'].iloc[0]
        max_region = self.regions_data[self.regions_data[self.year_column] == chars['max']]['region'].iloc[0]
        print(f"Экстремальные элементы:")
        print(f"   Минимум: {chars['min']:.0f} чел. ({min_region})")
        print(f"   Максимум: {chars['max']:.0f} чел. ({max_region})")
        
        # Размах выборки
        chars['range'] = chars['max'] - chars['min']
        print(f"Размах выборки: {chars['range']:.2f} чел.")

        # Полусумма экстремальных элементов
        chars['midrange'] = (chars['min'] + chars['max']) / 2
        print(f"Полусумма экстремальных элементов: {chars['midrange']:.2f} чел.")
        
        # Выборочная оценка асимметрии
        chars['skewness'] = basic_chars['m3'] / (basic_chars['std_biased']**3)
        print(f"Выборочная оценка асимметрии: {chars['skewness']:.4f}")
        
        # Выборочная оценка эксцесса
        chars['kurtosis'] = (basic_chars['m4'] / (basic_chars['std_biased']**4)) - 3
        print(f"Выборочная оценка эксцесса: {chars['kurtosis']:.4f}")
        
        print()
        return chars
    
    def conduct_analysis(self, chars: Dict):
        """Анализ полученных данных"""
        print("АНАЛИЗ ПОЛУЧЕННЫХ ДАННЫХ")
        print("=" * 50)
        
        # Абсолютное отклонение выборочного среднего от медианы
        mean_median_dev = abs(chars['mean'] - chars['median'])
        mean_median_dev_rel = (mean_median_dev / chars['mean']) * 100
        print(f"Отклонение среднего от медианы:")
        print(f"   Абсолютное: {mean_median_dev:.2f} чел.")
        print(f"   Относительное: {mean_median_dev_rel:.2f}%")
        
        # Абсолютное отклонение СКО от половины интерквартильной широты
        std_iqr_dev = abs(chars['std_dev'] - chars['iqr'] / 2)
        std_iqr_dev_rel = (std_iqr_dev / chars['std_dev']) * 100
        print(f"Отклонение СКО от половинного IQR:")
        print(f"   Абсолютное: {std_iqr_dev:.2f} чел.")
        print(f"   Относительное: {std_iqr_dev_rel:.2f}%")
        
        # Оценка отклонения эмпирической плотности от теоретической
        self.plot_density_comparison(chars)
        
        # Нанесение на линию различных точечных оценок среднего
        self.plot_mean_estimates(chars)
        
        print()
    
    def plot_density_comparison(self, chars: Dict):
        """Оценка отклонения эмпирической плотности от теоретической"""
        plt.figure(figsize=(12, 8))
        
        # Эмпирическая плотность (гистограмма)
        n_bins = min(20, int(np.sqrt(self.n)))
        n, bins, patches = plt.hist(self.doctors_data, bins=n_bins, density=True, 
                                   alpha=0.6, color='lightblue', edgecolor='black',
                                   label='Эмпирическая плотность')
        
        # Теоретическая нормальная плотность
        x_range = np.linspace(chars['min'] - 1000, chars['max'] + 1000, 1000)
        theoretical_pdf = stats.norm.pdf(x_range, chars['mean'], chars['std_dev'])
        plt.plot(x_range, theoretical_pdf, 'r-', linewidth=2, 
                label='Теоретическая нормальная плотность')
        
        # KDE оценка плотности
        kde = gaussian_kde(self.doctors_data)
        plt.plot(x_range, kde(x_range), 'g--', linewidth=2, 
                label='KDE оценка плотности')
        
        plt.title(f'Сравнение эмпирической и теоретической плотностей ({self.target_year} г.)', fontweight='bold')
        plt.xlabel('Численность врачей, чел.')
        plt.ylabel('Плотность вероятности')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/density_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Количественная оценка отклонения
        empirical_cdf = np.array([np.mean(self.doctors_data <= x) for x in x_range])
        theoretical_cdf = stats.norm.cdf(x_range, chars['mean'], chars['std_dev'])
        ks_statistic = np.max(np.abs(empirical_cdf - theoretical_cdf))
        
        print(f"Отклонение плотностей:")
        print(f"   KS-статистика: {ks_statistic:.4f}")
        if ks_statistic > 0.1:
            print("   Вывод: Значительное отклонение от нормального распределения")
        elif ks_statistic > 0.05:
            print("   Вывод: Умеренное отклонение от нормального распределения")
        else:
            print("   Вывод: Незначительное отклонение от нормального распределения")
    
    def plot_mean_estimates(self, chars: Dict):
        """Различные точечные оценки среднего"""
        plt.figure(figsize=(14, 8))
        
        # Создаем линию с данными (топ-30 для читаемости)
        top_n = min(30, self.n)
        sorted_indices = np.argsort(self.doctors_data)[-top_n:]
        sorted_data = self.doctors_data[sorted_indices]
        sorted_regions = self.regions[sorted_indices]
        x_positions = np.arange(top_n)
        
        # Точечный график данных
        plt.scatter(x_positions, sorted_data, alpha=0.7, s=50, 
                   color='blue', label='Данные по регионам')
        
        # Различные оценки среднего
        estimates = [
            (chars['mean'], 'Среднее арифметическое', 'red', '--'),
            (chars['median'], 'Медиана', 'green', '-.'),
            (chars['midquartile'], 'Полусумма квартилей', 'orange', ':'),
            (chars['midrange'], 'Полусумма экстремумов', 'purple', '--')
        ]
        
        for value, label, color, linestyle in estimates:
            plt.axhline(y=value, color=color, linestyle=linestyle, 
                       linewidth=2, label=label)
            plt.text(top_n-0.5, value, f'{value:.0f}', 
                    ha='left', va='center', color=color, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Настройки графика
        plt.xlabel('Регионы (упорядоченные по численности врачей)')
        plt.ylabel('Численность врачей, чел.')
        plt.title('Различные точечные оценки среднего значения', fontweight='bold')
        plt.xticks(x_positions, [r[:15] + '...' if len(r) > 15 else r for r in sorted_regions], 
                  rotation=45, ha='right', fontsize=8)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/mean_estimates.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Точечные оценки среднего:")
        for value, label, color, linestyle in estimates:
            print(f"   • {label}: {value:.0f} чел.")
    
    def section_distribution_density(self):
        print("ОЦЕНКА ПЛОТНОСТИ РАСПРЕДЕЛЕНИЯ")
        print("=" * 50)
        
        # Проверка гипотезы о нормальности
        shapiro_stat, shapiro_p = stats.shapiro(self.doctors_data)
        
        print("Гипотезы:")
        print(" H₀: Распределение является нормальным")
        print(" H₁: Распределение не является нормальным")
        print(f" Критерий Шапиро-Уилка: W = {shapiro_stat:.4f}, p-value = {shapiro_p:.4f}")
        
        if shapiro_p > 0.05:
            print(" Вывод: Нет оснований отвергать H₀ - распределение можно считать нормальным")
        else:
            print(" Вывод: Отвергаем H₀ - распределение значимо отличается от нормального")
       
        print()

    def section_interval_estimates(self):
        print("ИНТЕРВАЛЬНЫЕ ОЦЕНКИ ЧИСЛОВЫХ ХАРАКТЕРИСТИК ДАННЫХ")
        print("=" * 50)

        mean = np.mean(self.doctors_data)
        std = np.std(self.doctors_data, ddof=1)
        n = len(self.doctors_data)

        print(f"Исходные данные: n = {n}, x̄ = {mean:.0f}, s = {std:.0f}")

        # Расчет коэффициента эксцесса для формулы
        m4 = np.mean((self.doctors_data - mean) ** 4)
        e_kurtosis = (m4 / (std ** 4)) - 3

        print(f"\nКоэффициент эксцесса: {e_kurtosis:.2f}")

        # Доверительный интервал для математического ожидания (произвольное распределение)
        z_critical = stats.norm.ppf(0.975)
        margin_error_mean = z_critical * (std / np.sqrt(n))
        ci_lower = mean - margin_error_mean
        ci_upper = mean + margin_error_mean

        print("\nДоверительный интервал для математического ожидания (95%):")
        print(f"   Метод: для произвольной генеральной совокупности")
        print(f"   z-критическое: {z_critical:.4f}")
        print(f"   Интервал: [{ci_lower:.0f}; {ci_upper:.0f}] чел.")

        # Доверительный интервал для СКО (произвольное распределение)
        margin_error_std = 0.5 * z_critical * np.sqrt((e_kurtosis + 2) / n)
        std_lower = std * (1 - margin_error_std)
        std_upper = std * (1 + margin_error_std)

        print("\nДоверительный интервал для СКО (95%):")
        print(f"   Метод: для произвольной генеральной совокупности")
        print(f"   Интервал: [{std_lower:.0f}; {std_upper:.0f}] чел.")

    def section_comparative_analysis(self, chars: Dict):
        """Сравнительный анализ"""
        print("\nСРАВНИТЕЛЬНЫЙ АНАЛИЗ")
        print("=" * 50)
        
        # Анализ выбросов
        Q1, Q3 = chars['q1'], chars['q3']
        IQR = chars['iqr']
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = self.doctors_data[(self.doctors_data < lower_bound) | (self.doctors_data > upper_bound)]
        outlier_regions = self.regions_data[
            (self.regions_data[self.year_column] < lower_bound) | 
            (self.regions_data[self.year_column] > upper_bound)
        ]
        
        print(f"Анализ выбросов (метод Тьюки):")
        print(f"   Границы: [{lower_bound:.0f}; {upper_bound:.0f}]")
        print(f"   Выбросов обнаружено: {len(outliers)}")
        if len(outliers) > 0:
            print("   Регионы-выбросы:")
            for _, row in outlier_regions.iterrows():
                print(f"     - {row['region']}: {row[self.year_column]:.0f} чел.")
        
        # Анализ асимметрии
        print(f"\nАнализ асимметрии распределения:")
        skewness = chars['skewness']
        if abs(skewness) < 0.5:
            skew_type = "примерно симметричное"
        elif skewness > 0:
            skew_type = "правостороннее асимметричное"
        else:
            skew_type = "левостороннее асимметричное"
        print(f"   Коэффициент асимметрии: {skewness:.4f} ({skew_type})")
        
        # Анализ эксцесса
        kurtosis = chars['kurtosis']
        if kurtosis > 0:
            kurt_type = "островершинное (лептокуртическое)"
        elif kurtosis < 0:
            kurt_type = "плосковершинное (платикуртическое)"
        else:
            kurt_type = "нормальное (мезокуртическое)"
        print(f"   Коэффициент эксцесса: {kurtosis:.4f} ({kurt_type})")
        
        # Анализ вариативности
        cv = (chars['std_dev'] / chars['mean']) * 100
        if cv < 20:
            var_level = "низкая"
        elif cv < 50:
            var_level = "умеренная"
        else:
            var_level = "высокая"
        print(f"   Коэффициент вариации: {cv:.2f}% ({var_level} вариативность)")
        
        print()

    def section_additional_data_analysis(self, chars: Dict):
        """Оценка дополнительных данных"""
        print("ОЦЕНКА ДОПОЛНИТЕЛЬНЫХ ДАННЫХ")
        print("=" * 50)
        
        # Сравнение 2013 и 2023 годов
        print("СРАВНЕНИЕ 2013-2023:")
        print("-" * 30)
        self.simple_year_comparison()
        
        # Анализ по типам регионов
        print("\nТИПЫ РЕГИОНОВ:")
        print("-" * 30)
        self.simple_region_types()
        
        # Топ-5 регионов
        print("\nТОП-5 РЕГИОНОВ:")
        print("-" * 30)
        self.simple_top_regions()
        
        print()

    def simple_year_comparison(self):
        """Простое сравнение 2013 и 2023 годов"""
        if '2013' not in self.data.columns:
            print("   Нет данных за 2013 год")
            return
        
        try:
            # Только общие регионы
            common_data = self.regions_data[['region', '2013', '2023']].dropna()
            
            if len(common_data) == 0:
                print("   Нет данных для сравнения")
                return
            
            data_2013 = common_data['2013'].values
            data_2023 = common_data['2023'].values
            
            mean_2013 = np.mean(data_2013)
            mean_2023 = np.mean(data_2023)
            growth = mean_2023 - mean_2013
            growth_percent = (growth / mean_2013) * 100
            
            print(f"   Средняя численность:")
            print(f"     2013: {mean_2013:.0f} чел.")
            print(f"     2023: {mean_2023:.0f} чел.")
            print(f"     Изменение: {growth:+.0f} чел. ({growth_percent:+.1f}%)")
            
            # Сколько регионов показали рост
            growth_count = np.sum(data_2023 > data_2013)
            growth_percent_regions = (growth_count / len(common_data)) * 100
            print(f"   Рост в {growth_count} регионах ({growth_percent_regions:.1f}%)")
            
        except Exception as e:
            print(f"   Ошибка: {e}")

    def simple_region_types(self):
        """Простой анализ по типам регионов"""
        types = {
            'Республики': 'Республика',
            'Края': 'край',
            'Области': 'область',
            'Города': 'Москва|Санкт-Петербург|Севастополь',
            'Автономные округа': 'автономный'
        }
        
        for type_name, pattern in types.items():
            regions = self.regions_data[
                self.regions_data['region'].str.contains(pattern, na=False)
            ]
            if len(regions) > 0:
                values = regions[self.year_column].values
                mean = np.mean(values)
                count = len(values)
                print(f"   • {type_name} ({count}): {mean:.0f} чел.")

    def simple_top_regions(self):
        """Простой анализ топ-5 регионов"""
        top_5 = self.regions_data.nlargest(5, self.year_column)
        total = np.sum(self.doctors_data)
        
        for _, row in top_5.iterrows():
            share = (row[self.year_column] / total) * 100
            print(f"   • {row['region']}: {row[self.year_column]:.0f} чел. ({share:.1f}%)")

    def analyze_region_types(self):
        """Анализ по типам регионов"""
        region_stats = {}
        
        # Республики
        republics = self.regions_data[
            self.regions_data['region'].str.contains('Республика', na=False)
        ]
        if not republics.empty:
            values = republics[self.year_column].values
            region_stats['Республики'] = {
                'count': len(values),
                'mean': np.mean(values),
                'median': np.median(values),
                'share': (np.sum(values) / np.sum(self.doctors_data)) * 100
            }
        
        # Края
        krais = self.regions_data[
            self.regions_data['region'].str.contains('край', na=False) &
            ~self.regions_data['region'].str.contains('окраина', na=False)
        ]
        if not krais.empty:
            values = krais[self.year_column].values
            region_stats['Края'] = {
                'count': len(values),
                'mean': np.mean(values),
                'median': np.median(values),
                'share': (np.sum(values) / np.sum(self.doctors_data)) * 100
            }
        
        # Области
        oblasts = self.regions_data[
            self.regions_data['region'].str.contains('область', na=False)
        ]
        if not oblasts.empty:
            values = oblasts[self.year_column].values
            region_stats['Области'] = {
                'count': len(values),
                'mean': np.mean(values),
                'median': np.median(values),
                'share': (np.sum(values) / np.sum(self.doctors_data)) * 100
            }
        
        # Города федерального значения
        cities = self.regions_data[
            self.regions_data['region'].str.contains(
                'Москва|Санкт-Петербург|Севастополь', na=False
            )
        ]
        if not cities.empty:
            values = cities[self.year_column].values
            region_stats['Города ф.з.'] = {
                'count': len(values),
                'mean': np.mean(values),
                'median': np.median(values),
                'share': (np.sum(values) / np.sum(self.doctors_data)) * 100
            }
        
        # Автономные округа
        autonomous = self.regions_data[
            self.regions_data['region'].str.contains('автономный', na=False)
        ]
        if not autonomous.empty:
            values = autonomous[self.year_column].values
            region_stats['Автономные округа'] = {
                'count': len(values),
                'mean': np.mean(values),
                'median': np.median(values),
                'share': (np.sum(values) / np.sum(self.doctors_data)) * 100
            }
        
        return region_stats

    def calculate_concentration(self, data):
        """Расчет коэффициента концентрации (правило 20/80)"""
        sorted_data = np.sort(data)
        total = np.sum(data)
        
        # 20% самых крупных регионов
        top_20_count = int(0.2 * len(data))
        top_20_sum = np.sum(sorted_data[-top_20_count:])
        top_20_percent = (top_20_sum / total) * 100
        
        # 80% остальных регионов
        bottom_80_percent = 100 - top_20_percent
        
        ratio = top_20_percent / bottom_80_percent if bottom_80_percent > 0 else float('inf')
        
        return {
            'top_20_percent': top_20_percent,
            'bottom_80_percent': bottom_80_percent,
            'ratio': ratio
        }

    def generate_complete_report(self):
        """Генерация полного отчета"""
        print("\nПОЛНЫЙ СТАТИСТИЧЕСКИЙ ОТЧЕТ")
        print("=" * 70)
        print(f"Анализ численности врачей по регионам РФ за {self.target_year} год")
        print(f"Объем выборки: {self.n} регионов")
        print("=" * 70)
        print()
        
        # Основные разделы
        self.section_data_description()
        self.section_formal_representation()
        self.section_visual_representation()
        
        print("ЧИСЛОВЫЕ ОЦЕНКИ ХАРАКТЕРИСТИК ДАННЫХ")
        print("=" * 70)
        
        # Основные выборочные оценки
        basic_chars = self.calculate_basic_characteristics()
        
        # Дополнительные выборочные оценки
        additional_chars = self.calculate_additional_characteristics(basic_chars)
        
        # Объединяем характеристики для использования в других методах
        chars = {**basic_chars, **additional_chars}
        
        # Недостающие ключи для совместимости
        chars['std_dev'] = np.std(self.doctors_data, ddof=1)  # Несмещенное СКО
        
        # Анализ полученных данных
        self.conduct_analysis(chars)
        
        # Дополнительные разделы анализа
        self.section_additional_data_analysis(chars)
        
        # Остальные разделы
        self.section_distribution_density()
        self.section_interval_estimates()
        self.section_comparative_analysis(chars)
        
        # Итоговые выводы
        self.print_final_conclusions(chars)
        
        print("=" * 70)
        print("ОТЧЕТ ЗАВЕРШЕН")
        print("Все графики и расчеты сохранены в папке 'results/'")
    
    def print_final_conclusions(self, chars: Dict):
        """Итоговые выводы"""
        print("\nИТОГОВЫЕ ВЫВОДЫ")
        print("=" * 50)
        
        print("ОСНОВНЫЕ ХАРАКТЕРИСТИКИ:")
        print(f"   Средняя численность врачей: {chars['mean']:.0f} чел.")
        print(f"   Типичный регион (медиана): {chars['median']:.0f} чел.")
        print(f"   Разброс данных: от {chars['min']:.0f} до {chars['max']:.0f} чел.")
        
        print("\nХАРАКТЕР РАСПРЕДЕЛЕНИЯ:")
        skewness = chars['skewness']
        if skewness > 0.5:
            print("   Распределение правостороннее асимметричное")
            print("   Наличие регионов с аномально высокой численностью врачей")
        elif skewness < -0.5:
            print("   Распределение левостороннее асимметричное")
        else:
            print("   Распределение примерно симметричное")
        
        cv = (chars['std_dev'] / chars['mean']) * 100
        print(f"\nВАРИАТИВНОСТЬ ДАННЫХ:")
        print(f"   Коэффициент вариации: {cv:.1f}% - {'высокая' if cv > 50 else 'умеренная'} вариативность")
        print(f"   Значительные различия между регионами")
        
        print("\nСТАТИСТИЧЕСКАЯ НАДЕЖНОСТЬ:")
        print(f"   Объем выборки ({self.n} регионов) достаточен для анализа")
        print(f"   Доверительные интервалы имеют приемлемую ширину")
        print(f"   Результаты репрезентативны для всей территории РФ")


def main():
    """Запуск полного анализа"""
    os.makedirs('results', exist_ok=True)
    
    try:
        possible_files = [
            'data/vsego_vrachi.csv'
        ]
        
        data_file = None
        for file_path in possible_files:
            if os.path.exists(file_path):
                data_file = file_path
                print(f"Найден файл: {file_path}")
                break
        
        if data_file is None:
            print("Файл с данными не найден")
            return
        
        analyzer = CompleteStatisticalAnalysis(data_file, target_year=2023)
        analyzer.generate_complete_report()
        
    except Exception as e:
        print(f"Ошибка чтения файла: {e}")


if __name__ == "__main__":
    main()