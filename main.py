import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import split
from shapely.affinity import translate
from collections import Counter
from os.path import exists
from os import remove, mkdir
from shutil import rmtree
import numpy as np
from numpy import mean, std, sqrt, nan, median
from scipy.stats import shapiro
import xlsxwriter
import hdbscan
from sklearn.manifold import TSNE

from matplotlib import rc, cm
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage[utf8]{inputenc}')
rc('text.latex', preamble=r'\usepackage[russian]{babel}')

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', 1000)


class Processor:
    def __init__(self, **kwargs):
        self.__path_to_xlsx = kwargs['path_to_xlsx']
        self.__features_words = {}

        self.__path_to_all = 'all'
        self.__path_to_z, self.__path_to_y, self.__path_to_x = 'Z', 'Y', 'X'
        self.__k_all, self.__k_z, self.__k_y, self.__k_x = 0, 0, 0, 0

        self.__workbook = xlsxwriter.Workbook('results.xlsx')
        self.__silver_cell = self.__workbook.add_format({'bg_color': 'silver', 'bold': True})
        self.__bold_cell = self.__workbook.add_format({'bold': True})
        self.__worksheet_all, self.__worksheet_z, self.__worksheet_y, self.__worksheet_x = \
            self.__workbook.add_worksheet('Все поколения'), self.__workbook.add_worksheet('Поколение Z'), \
            self.__workbook.add_worksheet('Поколение Y'), self.__workbook.add_worksheet('Поколение X')
        self.__worksheet_all.set_column(0, 0, 200)
        self.__worksheet_z.set_column(0, 0, 200)
        self.__worksheet_y.set_column(0, 0, 200)
        self.__worksheet_x.set_column(0, 0, 200)

        self.__clear_previous()

        self.__base_columns = ['time', 'age', 'sex', 'education_level', 'education_profile', 'city']

        self.__n_questions = 134
        self.__n_statements1_min, self.__n_statements1_max = 7, 85
        self.__n_belief_min, self.__n_belief_max = 86, 90
        self.__n_statements2_min, self.__n_statements2_max = 91, 117
        self.__n_social_min, self.__n_social_max = 118, 134

        df_initial = self.__readout()
        self.__check(df_initial)

        #
        # preprocessing
        #
        df = self.__preprocess_cities(df_initial)
        df = self.__preprocess_answers(df)

        #
        # processing
        #

        # cities
        df_cities, n_top = self.__process_cities(df)
        self.__plot_cities_on_map(df_cities, n_top)

        # age
        self.__z_min, self.__z_max = 16, 21
        self.__y_min, self.__y_max = 22, 37
        self.__x_min, self.__x_max = 38, 51
        df_z, df_y, df_x = self.__process_age(df)

        for func in [self.__save_education_level_to_xlsx, self.__save_news_freq_to_xlsx,
                     self.__save_news_sources_to_xlsx, self.__save_news_sources_true_to_xlsx,
                     self.__save_news_sources_false_to_xlsx, self.__save_doubters_to_xlsx]:
            self.__k_all = func(**{'worksheet': self.__worksheet_all, 'df': df, 'k': self.__k_all})
            self.__k_z = func(**{'worksheet': self.__worksheet_z, 'df': df_z, 'k': self.__k_z})
            self.__k_y = func(**{'worksheet': self.__worksheet_y, 'df': df_y, 'k': self.__k_y})
            self.__k_x = func(**{'worksheet': self.__worksheet_x, 'df': df_x, 'k': self.__k_x})

        dfs = {'all': [], 'z': [], 'y': [], 'x': []}

        for name in dfs.keys():
            if name == 'all':
                df_cur, k, worksheet = df, self.__k_all, self.__worksheet_all
            elif name == 'z':
                df_cur, k, worksheet = df_z, self.__k_z, self.__worksheet_z
            elif name == 'y':
                df_cur, k, worksheet = df_y, self.__k_y, self.__worksheet_y
            elif name == 'x':
                df_cur, k, worksheet = df_x, self.__k_x, self.__worksheet_x

            dfs_st, df_belief, df_social = self.__split_by_questions(df_cur)
            for df_st in dfs_st:
                for pos_coeff in (True, False):
                    k = self.__calculate_core(df_st[1], label=df_st[0], worksheet=worksheet, k=k, positive_coeff=pos_coeff)
                    if name == 'all':
                        self.__k_all = k
                    elif name == 'z':
                        self.__k_z = k
                    elif name == 'y':
                        self.__k_y = k
                    else:
                        self.__k_x = k
                dfs[name] += [df_st, df_belief, df_social]

        # belief
        if exists('belief'):
            rmtree('belief')
        df_belief_mean, self.__k_all = self.__process_belief(self.__worksheet_all, dfs['all'][-2], self.__k_all, 'all')
        df_z_belief_mean, self.__k_z = self.__process_belief(self.__worksheet_z, dfs['z'][-2], self.__k_z, 'z')
        df_y_belief_mean, self.__k_y = self.__process_belief(self.__worksheet_y, dfs['y'][-2], self.__k_y, 'y')
        df_x_belief_mean, self.__k_x = self.__process_belief(self.__worksheet_x, dfs['x'][-2], self.__k_x, 'x')

        # trust
        if exists('trust'):
            rmtree('trust')
        df_social_mean, df_inst_mean, self.__k_all = self.__process_trust(self.__worksheet_all, dfs['all'][-1], self.__k_all, 'all')
        df_z_social_mean, df_z_inst_mean, self.__k_z = self.__process_trust(self.__worksheet_z, dfs['z'][-1], self.__k_z, 'z')
        df_y_social_mean, df_y_inst_mean, self.__k_y = self.__process_trust(self.__worksheet_y, dfs['y'][-1], self.__k_y, 'y')
        df_x_social_mean, df_x_inst_mean, self.__k_x = self.__process_trust(self.__worksheet_x, dfs['x'][-1], self.__k_x, 'x')

        # save excel for all
        df['belief_mean'] = df_belief_mean
        df['social_mean'] = df_social_mean
        df['inst_mean'] = df_inst_mean
        df.to_excel('data_output.xlsx')

        self.__clusterize(df)

        self.__workbook.close()

    def __clear_previous(self):
        if exists(self.__path_to_all):
            remove(self.__path_to_all)
        if exists(self.__path_to_z):
            remove(self.__path_to_z)
        if exists(self.__path_to_y):
            remove(self.__path_to_y)
        if exists(self.__path_to_x):
            remove(self.__path_to_x)

    @staticmethod
    def __check(df):
        # to_drop = []
        for feature in df.columns:
            col = df[feature]
            for i in range(len(col)):
                if col[i] is nan or col[i] is None or col[i] == '':
                    # to_drop.append((i, feature))
                    print(df.loc[i, :].head(50))
                    raise Exception('Empty cell in ({:03d}, {})'.format(i, feature))
        # print(to_drop)

    def __readout(self):
        columns = self.__base_columns + ['{:03d}'.format(i + 1) for i in range(self.__n_questions)]
        df = pd.read_excel(self.__path_to_xlsx)

        len_base = len(self.__base_columns)
        with open('features.txt', 'w') as f:
            for i in range(len_base, len(columns), 1):
                f.write('{}\n{}\n\n'.format(columns[i], df.columns[i]))
                self.__features_words.update({columns[i]: df.columns[i]})

        df.columns = columns

        return df

    def __save_process_age_to_xlsx(self, worksheet, n, n_male, n_female, k):
        worksheet.write(k, 0, 'ЧИСЛО РЕСПОНДЕНТОВ', self.__silver_cell); k += 1
        worksheet.write(k, 0, '{:03d}'.format(n)); k += 1
        worksheet.write(k, 0, 'ПОЛ', self.__silver_cell); k += 1
        worksheet.write(k, 0, 'Мужской: {:03d} ({:05.1f}%)'.format(n_male, n_male / n * 100)); k += 1
        worksheet.write(k, 0, 'Женский: {:03d} ({:05.1f}%)'.format(n_female, n_female / n * 100)); k += 1

        return k

    @staticmethod
    def __calculate_doubters(df):
        doubters = []
        for col_name in df.columns[12:]:
            n_doubters = len(df[(df[col_name] == 'Затрудняюсь ответить') | (df[col_name] == 3)])
            percent = n_doubters / len(df) * 100
            doubters.append((col_name, n_doubters, percent))

        return doubters

    def __save_doubters_to_xlsx(self, worksheet, df, k):
        doubters = self.__calculate_doubters(df)
        worksheet.write(k, 0, 'ЧИСЛО ЗАТРУДНИВШИХСЯ ОТВЕТИТЬ', self.__silver_cell); k += 1
        for i in range(len(doubters)):
            worksheet.write(k, 0, '({}) {}'.format(doubters[i][0], self.__features_words[doubters[i][0]]))
            worksheet.write(k, 1, '{:03d}'.format(doubters[i][1]))
            worksheet.write(k, 2, '{:05.1f}'.format(doubters[i][2])); k += 1

        return k

    def __save_education_level_to_xlsx(self, worksheet, df, k):
        n = len(df)
        n1, n2, n3, n4, n5 = len(df[df['education_level'] == 'Неполное среднее образование']), \
                             len(df[df['education_level'] == 'Среднее образование']), \
                             len(df[df['education_level'] == 'Среднее специальное образование']), \
                             len(df[df['education_level'] == 'Неполное высшее образование']), \
                             len(df[df['education_level'] == 'Высшее образование'])
        n1_percent, n2_percent, n3_percent, n4_percent, n5_percent = n1 / n * 100, n2 / n * 100, n3 / n * 100, \
                                                                     n4 / n * 100, n5 / n * 100
        worksheet.write(k, 0, 'УРОВЕНЬ ОБРАЗОВАНИЯ', self.__silver_cell); k += 1
        worksheet.write(k, 0, 'Неполное среднее образование: {:03d} ({:05.1f}%)'.format(n1, n1_percent)); k += 1
        worksheet.write(k, 0, 'Среднее образование: {:03d} ({:05.1f}%)'.format(n2, n2_percent)); k += 1
        worksheet.write(k, 0, 'Среднее специальное образование: {:03d} ({:05.1f}%)'.format(n3, n3_percent)); k += 1
        worksheet.write(k, 0, 'Неполное высшее образование: {:03d} ({:05.1f}%)'.format(n4, n4_percent)); k += 1
        worksheet.write(k, 0, 'Высшее образование: {:03d} ({:05.1f}%)'.format(n5, n5_percent)); k += 1

        return k

    def __save_news_freq_to_xlsx(self, worksheet, df, k):
        n = len(df)
        n1, n2, n3, n4, n5 = len(df[df['002'] == 'Много раз в день']), \
                             len(df[df['002'] == '1-2 раза в день']), \
                             len(df[df['002'] == 'Несколько раз в неделю']), \
                             len(df[df['002'] == 'Раз в неделю']), \
                             len(df[df['002'] == 'Реже'])
        n1_percent, n2_percent, n3_percent, n4_percent, n5_percent = n1 / n * 100, n2 / n * 100, n3 / n * 100, \
                                                                     n4 / n * 100, n5 / n * 100
        worksheet.write(k, 0, 'ЧАСТОТА ПРОСМОТРА НОВОСТЕЙ', self.__silver_cell); k += 1
        worksheet.write(k, 0, 'Много раз в день: {:03d} ({:05.1f}%)'.format(n1, n1_percent)); k += 1
        worksheet.write(k, 0, '1-2 раза в день: {:03d} ({:05.1f}%)'.format(n2, n2_percent)); k += 1
        worksheet.write(k, 0, 'Несколько раз в неделю: {:03d} ({:05.1f}%)'.format(n3, n3_percent)); k += 1
        worksheet.write(k, 0, 'Раз в неделю: {:03d} ({:05.1f}%)'.format(n4, n4_percent)); k += 1
        worksheet.write(k, 0, 'Реже: {:03d} ({:05.1f}%)'.format(n5, n5_percent)); k += 1

        return k

    def __save_news_sources_to_xlsx(self, worksheet, df, k):
        worksheet.write(k, 0, 'НОВОСТНЫЕ РЕСУРСЫ', self.__silver_cell); k += 1

        n = len(df)
        sources = [['Телевидение', 0, 0],
                   ['Паблики/группы в социальных сетях (VK, Facebook, Instagram, Twitter, Одноклассники и др.)', 0, 0],
                   ['Каналы в мессенджерах (телеграм, вайбер, вотсап)', 0, 0],
                   ['Новостные сайты', 0, 0],
                   ['Узнаю от знакомых/родственников/друзей', 0, 0],
                   ['Радио', 0, 0],
                   ['Газеты', 0, 0]]
        n_sources = len(sources)
        for s in range(n_sources):
            # look in df
            for i in range(n):
                answer = df.loc[i, '001']
                if sources[s][0] in answer:
                    sources[s][1] += 1
                    sources[s][2] = sources[s][1] / n * 100
            # save to xlsx
            worksheet.write(k, 0, '{}: {:03d} ({:05.1f}%)'.format(sources[s][0], sources[s][1], sources[s][2])); k += 1

        return k

    def __save_news_sources_true_to_xlsx(self, worksheet, df, k):
        worksheet.write(k, 0, 'НОВОСТНЫЕ РЕСУРСЫ, ЗАСЛУЖИВАЮЩИЕ ДОВЕРИЕ', self.__silver_cell); k += 1

        n = len(df)
        sources = [['Телевидение в целом', 0, 0],
                   ['Федеральные каналы по телевидению', 0, 0],
                   ['Интернет в целом', 0, 0],
                   ['Социальные сети', 0, 0],
                   ['Мессенджеры', 0, 0],
                   ['Новостные сайты', 0, 0],
                   ['Радио', 0, 0],
                   ['Газеты/журналы', 0, 0],
                   ['Ничего из вышеперечисленного', 0, 0]]
        n_sources = len(sources)
        for s in range(n_sources):
            # look in df
            for i in range(n):
                answer = df.loc[i, '005']
                if sources[s][0] in answer:
                    sources[s][1] += 1
                    sources[s][2] = sources[s][1] / n * 100
            # save to xlsx
            worksheet.write(k, 0, '{}: {:03d} ({:05.1f}%)'.format(sources[s][0], sources[s][1], sources[s][2])); k += 1

        return k

    def __save_news_sources_false_to_xlsx(self, worksheet, df, k):
        worksheet.write(k, 0, 'НОВОСТНЫЕ РЕСУРСЫ, НЕ ЗАСЛУЖИВАЮЩИЕ ДОВЕРИЕ', self.__silver_cell); k += 1

        n = len(df)
        sources = [['Телевидение в целом', 0, 0],
                   ['Федеральные каналы по телевидению', 0, 0],
                   ['Интернет в целом', 0, 0],
                   ['Социальные сети', 0, 0],
                   ['Мессенджеры', 0, 0],
                   ['Новостные сайты', 0, 0],
                   ['Радио', 0, 0],
                   ['Газеты/журналы', 0, 0],
                   ['Ничего из вышеперечисленного', 0, 0]]
        n_sources = len(sources)
        for s in range(n_sources):
            # look in df
            for i in range(n):
                answer = df.loc[i, '006']
                if sources[s][0] in answer:
                    sources[s][1] += 1
                    sources[s][2] = sources[s][1] / n * 100
            # save to xlsx
            worksheet.write(k, 0, '{}: {:03d} ({:05.1f}%)'.format(sources[s][0], sources[s][1], sources[s][2])); k += 1

        return k

    def __process_age(self, df):
        n = len(df)

        df_z = df.loc[(df['age'] >= self.__z_min) & (df['age'] <= self.__z_max)]
        df_y = df.loc[(df['age'] >= self.__y_min) & (df['age'] <= self.__y_max)]
        df_x = df.loc[(df['age'] >= self.__x_min) & (df['age'] <= self.__x_max)]
        df_z, df_y, df_x = df_z.reset_index(drop=True), df_y.reset_index(drop=True), df_x.reset_index(drop=True)
        n_z, n_y, n_x = len(df_z), len(df_y), len(df_x)

        df_male = df.loc[(df['sex'] == 'Мужской')]
        df_female = df.loc[(df['sex'] == 'Женский')]
        n_male, n_female = len(df_male), len(df_female)

        df_z_male, df_z_female = df_z.loc[(df_z['sex'] == 'Мужской')], df_z.loc[(df_z['sex'] == 'Женский')]
        df_y_male, df_y_female = df_y.loc[(df_y['sex'] == 'Мужской')], df_y.loc[(df_y['sex'] == 'Женский')]
        df_x_male, df_x_female = df_x.loc[(df_x['sex'] == 'Мужской')], df_x.loc[(df_x['sex'] == 'Женский')]

        n_z_male, n_z_female = len(df_z_male), len(df_z_female)
        n_y_male, n_y_female = len(df_y_male), len(df_y_female)
        n_x_male, n_x_female = len(df_x_male), len(df_x_female)

        self.__k_all = self.__save_process_age_to_xlsx(self.__worksheet_all, n, n_male, n_female, self.__k_all)
        self.__worksheet_all.write(self.__k_all, 0, 'ВОЗРАСТ', self.__silver_cell); self.__k_all += 1
        self.__worksheet_all.write(self.__k_all, 0, 'Z: {:03d} ({:05.1f}%)'.format(n_z, n_z / n * 100)); self.__k_all += 1
        self.__worksheet_all.write(self.__k_all, 0, 'Y: {:03d} ({:05.1f}%)'.format(n_y, n_y / n * 100)); self.__k_all += 1
        self.__worksheet_all.write(self.__k_all, 0, 'X: {:03d} ({:05.1f}%)'.format(n_x, n_x / n * 100)); self.__k_all += 1

        self.__k_z = self.__save_process_age_to_xlsx(self.__worksheet_z, n_z, n_z_male, n_z_female, self.__k_z)
        self.__k_y = self.__save_process_age_to_xlsx(self.__worksheet_y, n_y, n_y_male, n_y_female, self.__k_y)
        self.__k_x = self.__save_process_age_to_xlsx(self.__worksheet_x, n_x, n_x_male, n_x_female, self.__k_x)

        return df_z, df_y, df_x

    @staticmethod
    def __preprocess_cities(df):
        # filter and aggregate cities
        i_to_drop = []
        for i in range(len(df['city'])):
            if isinstance(df.loc[i, 'city'], (int, float)):
                i_to_drop.append(i)
                continue
            city = df.loc[i, 'city'].lstrip().rstrip()
            if city in ('москва', 'Филадельфия', 'Дублин', 'Париж', 'Зеленоград',
                        'Ваймар', 'Хельсинки', 'Алматы', 'сасква', 'Алма-Ата',
                        'Москва, последний год - Тульская область, г Суворов. (мб можно его указать, чтобы не все были из мск))) )',
                        'страна Казахстан, город Атырау', 'Бостон, США', 'Moscow', 'Италия'):
                city = 'Москва'
            elif city in ('Одинцово', 'Люберцы', 'Ступино', 'Ступино московской обл',
                          'Мытищи', 'Химки', 'Балашиха', 'Ступино Московской области',
                          'Г. Троицк, Москва', 'Хотьково', 'Пушкино', 'Железнодорожный',
                          'Ступино Московской обл', 'Серпухов', 'Котельники', 'Сергиев Посад'):
                city = 'Московская область'
            elif city in ('Волгодонск', 'Волгодоск', 'Ростовская обл', 'Живу в деревне', 'Азов', 'Волгодонск Ростовская обл.',
                          'Ростов-на-Дону', 'Ростов на Дону'):
                city = 'Ростовская область'
            elif city in ('питер', 'Ереван', 'Санкт петеобург', 'Санкт-Перетбург', 'СПб', 'Вентспилс', 'Волхов', 'Минск'):
                city = 'Санкт-Петербург'
            elif city in ('райцентр Курской области'):
                city = 'Курск'
            elif city in ('вологда', 'Череповец'):
                city = 'Вологда'
            elif city in ('Кишинев', 'Оренбург'):
                city = 'Екатеринбург'
            elif city in ('Березники(Пермский край)'):
                city = 'Пермь'
            elif city in ('Усолье-Сибирское (Иркутская область)'):
                city = 'Иркутск'
            elif city in ('Самарская область,с Челно-вершины', 'Ульяновск'):
                city = 'Самара'
            elif city in ('ВОЛГОГРАД'):
                city = 'Волгоград'
            elif city in ('Суворов'):
                city = 'Тула'
            elif city in ('Ростов'):
                city = 'Ярославль'
            elif city in ('Железногорск'):
                city = 'Красноярск'
            elif city in ('Краснодар', 'Невинномысск', 'Севастополь', 'Бишкек', 'Пятигорск', 'Владикавказ'):
                city = 'Краснодарский край'
            elif city in ('Г. Балашов'):
                city = 'Саратов'
            elif city in ('Усть-Кан', 'Горно-Алтайск', 'Барнаул'):
                city = 'Новосибирск'
            elif city in ('Северодвинск'):
                city = 'Архангельск'
            elif city in ('Северодвинск'):
                city = 'Архангельск'
            elif city in ('Елец'):
                city = 'Воронеж'

            df.loc[i, 'city'] = city
        df = df.drop(df.index[i_to_drop]).reset_index(drop=True)

        return df

    @staticmethod
    def __process_cities(df):
        counts = Counter(df['city'])
        df_cities = pd.DataFrame.from_dict(counts, orient='index', columns=['number'])
        df_cities['percent'] = df_cities['number'] / len(df) * 100
        df_cities = df_cities.sort_values(by=['number'], ascending=True)

        cities = df_cities.index.values.tolist()
        coords = {'Москва': (37.6173, 55.7558),
                  'Московская область': (37.6173, 55.7558),
                  'Санкт-Петербург': (30.3609, 59.9311),
                  'Ростовская область': (39.7015, 47.2357),
                  'Краснодарский край': (38.9746, 45.0360),
                  'Екатеринбург': (60.6454, 56.8431),
                  'Волгоград': (44.5133, 48.7080),
                  'Тула': (44.5133, 48.7080),
                  'Горно-Алтайск': (85.9601, 51.9587),
                  'Саратов': (46.0154, 51.5462),
                  'Новосибирск': (82.8964, 54.9833),
                  'Самара': (50.1606, 53.2038),
                  'Нижний Новгород': (44.0059, 56.3269),
                  'Курск': (36.1947, 51.7482),
                  'Ярославль': (39.8845, 57.6261),
                  'Смоленск': (32.0504, 54.7903),
                  'Белгород': (36.5983, 50.5997),
                  'Вологда': (39.8978, 59.2181),
                  'Пермь': (56.2270, 58.0092),
                  'Красноярск': (92.8932, 56.0153),
                  'Калуга': (36.2637, 54.5136),
                  'Воронеж': (39.1919, 51.6683),
                  'Элиста': (44.2794, 46.3155),
                  'Брянск': (34.4161, 53.2635),
                  'Архангельск': (40.5506, 64.5459),
                  'Кемерово': (86.0621, 55.3443),
                  'Великий Новгород': (31.2742, 58.5256),
                  'Казань': (49.1233, 55.7879),
                  'Усть-Кан': (84.7522, 50.9332),
                  'Йошкар-Ола': (47.8896, 56.6418),
                  'Пенза': (45.0000, 53.2273),
                  'Иркутск': (104.2890, 52.2855),
                  'Владимир': (40.3896, 56.1428),
                  'Мурманск': (33.0856, 68.9733),
                  'Омск': (73.3645, 54.9914),
                  'Барнаул': (83.7836, 53.3497),
                  'Тюмень': (65.5619, 57.1553)
                  }
        for city in cities:
            df_cities.loc[city, 'lon'] = coords[city][0]
            df_cities.loc[city, 'lat'] = coords[city][1]

        # plot number
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()
        df_cities['number'].plot.barh(color='black', legend=False)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Число респондентов', fontsize=20)
        plt.grid(c='gray', lw=1, ls=':', alpha=0.5)
        ax.set_axisbelow(True)
        plt.savefig('cities_number.png', bbox_inches='tight', dpi=200)
        plt.close()

        # plot percent
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()
        df_cities['percent'].plot.barh(color='red', legend=False)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Число респондентов, $\%$', fontsize=20)
        plt.grid(c='gray', lw=1, ls=':', alpha=0.5)
        ax.set_axisbelow(True)
        plt.savefig('cities_percent.png', bbox_inches='tight', dpi=200)
        plt.close()

        # select only top
        n_top = 5
        cc = [(k, v) for k, v in counts.items()]
        cc.sort(key=lambda x: x[1], reverse=True)
        cc_agg = cc[:n_top]
        n_other = 0
        for i in range(n_top, len(cc)):
            n_other += cc[i][1]
        cc_agg.append(('Другие города', n_other))
        counts_agg = dict(cc_agg)

        # make dataframe
        df_cities_agg = pd.DataFrame.from_dict(counts_agg, orient='index', columns=['number'])
        df_cities_agg['percent'] = df_cities_agg['number'] / len(df) * 100
        df_cities_agg = df_cities_agg.sort_values(by=['number'], ascending=True)

        # plot number aggregated
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()
        df_cities_agg['number'].plot.barh(color='black', legend=False)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Число респондентов', fontsize=20)
        plt.grid(c='gray', lw=1, ls=':', alpha=0.5)
        ax.set_axisbelow(True)
        plt.savefig('cities_number_aggregated.png', bbox_inches='tight', dpi=200)
        plt.close()

        # plot percent aggregated
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot()
        df_cities_agg['percent'].plot.barh(color='red', legend=False)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Число респондентов, $\%$', fontsize=20)
        plt.grid(c='gray', lw=1, ls=':', alpha=0.5)
        ax.set_axisbelow(True)
        plt.savefig('cities_percent_aggregated.png', bbox_inches='tight', dpi=200)
        plt.close()

        return df_cities, n_top

    @staticmethod
    def __plot_cities_on_map(df, n_top):
        def shift_geom(shift, gdataframe, plotQ=False):
            # this code is adapted from answer found in SO
            # will be credited here: ???
            shift -= 180
            moved_geom = []
            splitted_geom = []
            border = LineString([(shift, 90), (shift, -90)])

            for row in gdataframe["geometry"]:
                splitted_geom.append(split(row, border))
            for element in splitted_geom:
                items = list(element)
                for item in items:
                    minx, miny, maxx, maxy = item.bounds
                    if minx >= shift:
                        moved_geom.append(translate(item, xoff=-180 - shift))
                    else:
                        moved_geom.append(translate(item, xoff=180 - shift))

            # got `moved_geom` as the moved geometry
            moved_geom_gdf = gpd.GeoDataFrame({"geometry": moved_geom})

            # can change crs here
            if plotQ:
                fig1, ax1 = plt.subplots(figsize=[8, 6])
                moved_geom_gdf.plot(ax=ax1)
                plt.show()

            return moved_geom_gdf

        df = df.sort_values(by=['number'], ascending=False)

        # delete Московская область
        df = df.drop(df.index[1])

        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        russia = shift_geom(-90, shift_geom(90, world[world['iso_a3'] == 'RUS'], False), True)
        russia.plot(figsize=(20, 20))

        for i, city in enumerate(df.index.values.tolist()):
            s = 200 if i < n_top - 1 else 20
            plt.scatter(df.loc[city, 'lon'], df.loc[city, 'lat'], c='red', s=s, alpha=1)
            if i < n_top - 1:
                city_label = city
                if city == 'Ростовская область':
                    city_label = 'Ростов-на-Дону'
                elif city == 'Краснодарский край':
                    city_label = 'Краснодар'
                plt.text(df.loc[city, 'lon'] + 1, df.loc[city, 'lat'], city_label, color='black',
                         fontsize=20, zorder=1)
        plt.xticks([])
        plt.yticks([])
        plt.xlim([22, 110])
        plt.savefig('map.png', bbox_inches='tight', dpi=200)

    def __split_by_questions(self, df):
        # statements1 = ['{:03d}'.format(i) for i in range(self.__n_statements1_min, self.__n_statements1_max + 1, 1)]
        # statements2 = ['{:03d}'.format(i) for i in range(self.__n_statements2_min, self.__n_statements2_max + 1, 1)]

        st1 = ['Оценка', ['{:03d}'.format(i) for i in range(7, 15, 1)] + ['{:03d}'.format(i) for i in range(22, 24, 1)]]
        st2 = ['Причины', ['{:03d}'.format(i) for i in range(50, 75, 1)] + ['{:03d}'.format(i) for i in range(83, 86, 1)]]
        st3 = ['Отношения', ['{:03d}'.format(i) for i in range(36, 46, 1)] + ['047'] + ['091', '092']]
        st4 = ['Критерии', ['{:03d}'.format(i) for i in range(93, 118, 1)]]
        st5 = ['Про другие поколения', ['{:03d}'.format(i) for i in range(15, 22, 1)]]
        st6 = ['Прогноз', ['{:03d}'.format(i) for i in range(24, 36, 1)]]
        st7 = ['Действие', ['046'] + ['049'] + ['{:03d}'.format(i) for i in range(75, 83, 1)]]
        st_all = ['Утверждения', st1[1] + st2[1] + st3[1] + st4[1] + st5[1], st6[1], st7[1]]
        sts = [st_all, st1, st2, st3, st4, st5, st6, st7]

        belief = ['{:03d}'.format(i) for i in range(self.__n_belief_min, self.__n_belief_max + 1, 1)]
        social = ['{:03d}'.format(i) for i in range(self.__n_social_min, self.__n_social_max + 1, 1)]

        dfs_st = []
        for st in sts:
            df_selected = df[st[1]]
            dfs_st.append((st[0], df_selected))

        df_belief, df_social = df[belief], df[social]

        # df_statements1, df_statements2, df_statements, df_belief, df_social = \
        #     df[statements1], df[statements2], df[statements1 + statements2], df[belief], df[social]

        return dfs_st, df_belief, df_social

    def __calculate_core(self, df, label, worksheet, k, positive_coeff=True):
        if positive_coeff:
            target_answers = (1, 2)
        else:
            target_answers = (4, 5)

        features = list(df.columns)
        n_features = len(features)
        coeff_pos = []
        n = len(df)
        for feature in features:
            n_pos = 0
            for i in range(n):
                if df.loc[i, feature] in target_answers:
                    n_pos += 1
            coeff_pos.append(n_pos / n * 100)
        mean_coeff = mean(coeff_pos)
        sigma_coeff = sqrt(std(coeff_pos))
        border = mean_coeff + sigma_coeff

        core, precore, periphery = [], [], []
        for i in range(n_features):
            if coeff_pos[i] > border:
                core.append((features[i], coeff_pos[i]))
            elif mean_coeff <= coeff_pos[i] <= border:
                precore.append((features[i], coeff_pos[i]))
            else:
                periphery.append((features[i], coeff_pos[i]))
        core.sort(key=lambda x: x[1], reverse=True)
        precore.sort(key=lambda x: x[1], reverse=True)
        periphery.sort(key=lambda x: x[1], reverse=True)

        if positive_coeff:
            worksheet.write(k, 0, 'КОЭФФИЦИЕНТЫ ПОЗИТИВНЫХ ОТВЕТОВ: {}'.format(label), self.__silver_cell); k += 1
        else:
            worksheet.write(k, 0, 'КОЭФФИЦИЕНТЫ НЕГАТИВНЫХ ОТВЕТОВ: {}'.format(label), self.__silver_cell); k += 1
        worksheet.write(k, 0, 'Число утверждений: {:03d}'.format(n_features)); k += 1
        for i in range(len(features)):
            worksheet.write(k, 0, '({}) {}'.format(features[i], self.__features_words[features[i]]))
            worksheet.write(k, 1, '{:05.1f}'.format(coeff_pos[i]));
            k += 1
        worksheet.write(k, 0, 'Математическое ожидание M: {:05.1f}'.format(mean_coeff)); k += 1
        worksheet.write(k, 0, 'Стандартное отклонение s: {:05.1f}'.format(sigma_coeff)); k += 1
        worksheet.write(k, 0, 'Граница M + s: {:05.1f}'.format(mean_coeff + sigma_coeff)); k += 1
        worksheet.write(k, 0, 'ЯДРО', self.__bold_cell); k += 1
        worksheet.write(k, 0, 'Число утверждений: {:03d} ({:05.1f}%)'.format(len(core), len(core) / n_features * 100)); k += 1
        for i in range(len(core)):
            worksheet.write(k, 0, '({}) {}'.format(core[i][0], self.__features_words[core[i][0]]))
            worksheet.write(k, 1, '{:05.1f}'.format(core[i][1])); k += 1
        worksheet.write(k, 0, 'ПЕРИФЕРИЯ, БЛИЗКАЯ К ЯДРУ', self.__bold_cell); k += 1
        worksheet.write(k, 0, 'Число утверждений: {:03d} ({:05.1f}%)'.format(len(precore), len(precore) / n_features * 100)); k += 1
        for i in range(len(precore)):
            worksheet.write(k, 0, '({}) {}'.format(precore[i][0], self.__features_words[precore[i][0]]))
            worksheet.write(k, 1, '{:05.1f}'.format(precore[i][1])); k += 1
        worksheet.write(k, 0, 'ПЕРИФЕРИЯ', self.__bold_cell); k += 1
        worksheet.write(k, 0, 'Число утверждений: {:03d} ({:05.1f}%)'.format(len(periphery), len(periphery) / n_features * 100)); k += 1
        for i in range(len(periphery)):
            worksheet.write(k, 0, '({}) {}'.format(periphery[i][0], self.__features_words[periphery[i][0]]))
            worksheet.write(k, 1, '{:05.1f}'.format(periphery[i][1])); k += 1
        return k

    @staticmethod
    def __calculate_levels(df, min_val, max_val):
        mean_val = 0.5 * (min_val + max_val)
        delta = max_val - min_val
        val_15, val_25, val_75, val_85 = 0.15 * delta + min_val, 0.25 * delta + min_val, 0.75 * delta + min_val, 0.85 * delta + min_val
        n_0_15 = len(df[df < val_15])
        n_15_25 = len(df[(df > val_15) & (df < val_25)])
        n_25_50 = len(df[(df > val_25) & (df < mean_val)])
        n_50_75 = len(df[(df > mean_val) & (df < val_75)])
        n_75_85 = len(df[(df > val_75) & (df < val_85)])
        n_85_100 = len(df[df > val_85])

        return n_0_15, n_15_25, n_25_50, n_50_75, n_75_85, n_85_100

    @staticmethod
    def __preprocess_answers(df):
        df = df.replace('Абсолютно согласен', 1)
        df = df.replace('Согласен', 2)
        df = df.replace('Затрудняюсь ответить', 3)
        df = df.replace('Не согласен', 4)
        df = df.replace('Абсолютно не согласен', 5)

        df = df.replace('Полностью согласен', 1)
        df = df.replace('Скорее согласен', 2)
        df = df.replace('Затрудняюсь ответить', 3)
        df = df.replace('Скорее не согласен', 4)
        df = df.replace('Полностью не согласен', 5)

        return df

    def __process_belief(self, worksheet, df, k, label):
        inversed = [86, 87, 90]
        for i in inversed:
            name = '{:03d}'.format(i)
            df.loc[:][name].replace([1, 2, 4, 5], [5, 4, 2, 1], inplace=True)
        df_mean = df.mean(numeric_only=True, axis=1)

        n = len(df_mean)

        m_0_15, m_15_25, m_25_50, m_50_75, m_75_85, m_85_100 = self.__calculate_levels(df_mean, 1, 5)
        m_0_15_rel, m_15_25_rel, m_25_50_rel, m_50_75_rel, m_75_85_rel, m_85_100_rel \
            = m_0_15 / n * 100, m_15_25 / n * 100, m_25_50 / n * 100, m_50_75 / n * 100, m_75_85 / n * 100, m_85_100 / n * 100

        n_0_15, n_15_25, n_25_50, n_50_75, n_75_85, n_85_100 = self.__calculate_levels(df_mean, np.min(df_mean), np.max(df_mean))
        n_0_15_rel, n_15_25_rel, n_25_50_rel, n_50_75_rel, n_75_85_rel, n_85_100_rel \
            = n_0_15 / n * 100, n_15_25 / n * 100, n_25_50 / n * 100, n_50_75 / n * 100, n_75_85 / n * 100, n_85_100 / n * 100

        # Shapiro-Wilk
        stat, p = shapiro(df_mean)

        # save to xlsx
        worksheet.write(k, 0, 'ВЕРА В ЛЮДЕЙ', self.__silver_cell); k += 1
        worksheet.write(k, 0, 'Критерий Шапиро-Уилка: stat = {:05.3f}, p = {:05.3f}'.format(stat, p)); k += 1
        worksheet.write(k, 0, 'Уровни (по методике):'); k += 1
        worksheet.write(k, 0, '000-015%: {:03d} ({:05.1f}%)'.format(m_0_15, m_0_15_rel)); k += 1
        worksheet.write(k, 0, '015-025%: {:03d} ({:05.1f}%)'.format(m_15_25, m_15_25_rel)); k += 1
        worksheet.write(k, 0, '025-050%: {:03d} ({:05.1f}%)'.format(m_25_50, m_25_50_rel)); k += 1
        worksheet.write(k, 0, '050-075%: {:03d} ({:05.1f}%)'.format(m_50_75, m_50_75_rel)); k += 1
        worksheet.write(k, 0, '075-085%: {:03d} ({:05.1f}%)'.format(m_75_85, m_75_85_rel)); k += 1
        worksheet.write(k, 0, '085-100%: {:03d} ({:05.1f}%)'.format(m_85_100, m_85_100_rel)); k += 1
        worksheet.write(k, 0, 'Уровни (по выборке):'); k += 1
        worksheet.write(k, 0, '000-015%: {:03d} ({:05.1f}%)'.format(n_0_15, n_0_15_rel)); k += 1
        worksheet.write(k, 0, '025-050%: {:03d} ({:05.1f}%)'.format(n_15_25, n_15_25_rel)); k += 1
        worksheet.write(k, 0, '025-050%: {:03d} ({:05.1f}%)'.format(n_25_50, n_25_50_rel)); k += 1
        worksheet.write(k, 0, '050-075%: {:03d} ({:05.1f}%)'.format(n_50_75, n_50_75_rel)); k += 1
        worksheet.write(k, 0, '075-085%: {:03d} ({:05.1f}%)'.format(n_75_85, n_75_85_rel)); k += 1
        worksheet.write(k, 0, '085-100%: {:03d} ({:05.1f}%)'.format(n_85_100, n_85_100_rel)); k += 1

        dirname = 'belief'
        if not exists(dirname):
            mkdir(dirname)

        with open('{}/statistics.txt'.format(dirname), 'a') as f:
            f.write('{}:\n'.format(label))
            f.write('mean = {:05.1f}\n'.format(mean(df_mean)))
            f.write('median = {:05.1f}\n'.format(median(df_mean)))

        df.to_excel('{}/belief.xlsx'.format(dirname))
        df_mean.to_excel('{}/belief_mean.xlsx'.format(dirname))

        plt.figure(figsize=(10, 10))
        df_mean.plot.hist(bins=10, color='black', legend=False)
        plt.xlim([1, 5])
        plt.savefig('{}/{}_mean.png'.format(dirname, label), bbox_inches='tight', dpi=200)
        plt.close()

        return df_mean, k

    def __process_trust_essence(self, worksheet, df_mean, dirname, part_name, label, essence_name, k):
        # calculate
        n = len(df_mean)

        m_0_15, m_15_25, m_25_50, m_50_75, m_75_85, m_85_100 = self.__calculate_levels(df_mean, 1, 5)
        m_0_15_rel, m_15_25_rel, m_25_50_rel, m_50_75_rel, m_75_85_rel, m_85_100_rel \
            = m_0_15 / n * 100, m_15_25 / n * 100, m_25_50 / n * 100, m_50_75 / n * 100, m_75_85 / n * 100, m_85_100 / n * 100

        n_0_15, n_15_25, n_25_50, n_50_75, n_75_85, n_85_100 = self.__calculate_levels(df_mean, np.min(df_mean),
                                                                                       np.max(df_mean))
        n_0_15_rel, n_15_25_rel, n_25_50_rel, n_50_75_rel, n_75_85_rel, n_85_100_rel \
            = n_0_15 / n * 100, n_15_25 / n * 100, n_25_50 / n * 100, n_50_75 / n * 100, n_75_85 / n * 100, n_85_100 / n * 100

        # Shapiro-Wilk
        stat, p = shapiro(df_mean)

        # save to xlsx
        worksheet.write(k, 0, '{}'.format(part_name), self.__silver_cell); k += 1
        worksheet.write(k, 0, 'Критерий Шапиро-Уилка: stat = {:05.3f}, p = {:05.3f}'.format(stat, p)); k += 1
        worksheet.write(k, 0, 'Уровни (по методике):'); k += 1
        worksheet.write(k, 0, '000-015%: {:03d} ({:05.1f}%)'.format(m_0_15, m_0_15_rel)); k += 1
        worksheet.write(k, 0, '015-025%: {:03d} ({:05.1f}%)'.format(m_15_25, m_15_25_rel)); k += 1
        worksheet.write(k, 0, '025-050%: {:03d} ({:05.1f}%)'.format(m_25_50, m_25_50_rel)); k += 1
        worksheet.write(k, 0, '050-075%: {:03d} ({:05.1f}%)'.format(m_50_75, m_50_75_rel)); k += 1
        worksheet.write(k, 0, '075-085%: {:03d} ({:05.1f}%)'.format(m_75_85, m_75_85_rel)); k += 1
        worksheet.write(k, 0, '085-100%: {:03d} ({:05.1f}%)'.format(m_85_100, m_85_100_rel)); k += 1
        worksheet.write(k, 0, 'Уровни (по выборке):'); k += 1
        worksheet.write(k, 0, '000-015%: {:03d} ({:05.1f}%)'.format(n_0_15, n_0_15_rel)); k += 1
        worksheet.write(k, 0, '025-050%: {:03d} ({:05.1f}%)'.format(n_15_25, n_15_25_rel)); k += 1
        worksheet.write(k, 0, '025-050%: {:03d} ({:05.1f}%)'.format(n_25_50, n_25_50_rel)); k += 1
        worksheet.write(k, 0, '050-075%: {:03d} ({:05.1f}%)'.format(n_50_75, n_50_75_rel)); k += 1
        worksheet.write(k, 0, '075-085%: {:03d} ({:05.1f}%)'.format(n_75_85, n_75_85_rel)); k += 1
        worksheet.write(k, 0, '085-100%: {:03d} ({:05.1f}%)'.format(n_85_100, n_85_100_rel)); k += 1

        with open('{}/{}_statistics.txt'.format(dirname, essence_name), 'a') as f:
            f.write('{}:\n'.format(label))
            f.write('mean = {:05.1f}\n'.format(mean(df_mean)))
            f.write('median = {:05.1f}\n'.format(median(df_mean)))

        # plot
        plt.figure(figsize=(10, 10))
        df_mean.plot.hist(bins=10, color='black', legend=False)
        plt.xlim([1, 5])
        plt.savefig('{}/{}_{}_mean.png'.format(dirname, essence_name, label), bbox_inches='tight', dpi=200)
        plt.close()

        return k

    def __process_trust(self, worksheet, df, k, label):
        direct = [118, 121, 122, 123, 125, 127, 129, 133]
        for i in direct:
            name = '{:03d}'.format(i)
            df.loc[:][name].replace([1, 2, 4, 5], [5, 4, 2, 1], inplace=True)

        # make social dir
        dirname = 'trust'
        if not exists(dirname):
            mkdir(dirname)

        #
        # social trust
        #

        social = ['{:03d}'.format(i + 118) for i in range(17)]
        df_social = df[social]
        df_social_mean = df_social.mean(numeric_only=True, axis=1)
        df_social.to_excel('{}/social.xlsx'.format(dirname))
        df_social_mean.to_excel('{}/social_mean.xlsx'.format(dirname))

        k = self.__process_trust_essence(worksheet, df_social_mean, dirname, 'СОЦИАЛЬНОЕ ДОВЕРИЕ', label, 'social', k)

        #
        # inst trust
        #

        inst = ['118', '121', '123', '125', '126', '128', '129', '131', '132']
        df_inst = df[inst]
        df_inst_mean = df_inst.mean(numeric_only=True, axis=1)
        df_inst.to_excel('{}/inst.xlsx'.format(dirname))
        df_inst_mean.to_excel('{}/inst_mean.xlsx'.format(dirname))

        k = self.__process_trust_essence(worksheet, df_inst_mean, dirname, 'ИНСТИТУЦИОНАЛЬНОЕ ДОВЕРИЕ', label, 'inst', k)

        return df_social_mean, df_inst_mean, k

    @staticmethod
    def __normalize_scale_1_to_5(col):
        col = 2 * ((col - 1) / 4 - 0.5)

        return col

    def __clusterize(self, df):
        #
        # preprocessing and normalization
        #

        # delete some columns
        df = df.drop(['time', 'education_profile', 'city', '001', '003', '004', '005', '006'], axis=1)

        # age
        min_age, max_age = np.min(df['age']), np.max(df['age'])
        df['age'] = 2 * ((df['age'] - min_age) / (max_age - min_age) - 0.5)

        # sex
        df['sex'] = df['sex'].replace('Женский', -1)
        df['sex'] = df['sex'].replace('Мужской', 1)

        # education_level
        df['education_level'] = df['education_level'].replace('Неполное среднее образование', 1)
        df['education_level'] = df['education_level'].replace('Среднее образование', 2)
        df['education_level'] = df['education_level'].replace('Среднее специальное образование', 3)
        df['education_level'] = df['education_level'].replace('Неполное высшее образование', 4)
        df['education_level'] = df['education_level'].replace('Высшее образование', 5)
        df['education_level'] = self.__normalize_scale_1_to_5(df['education_level'])

        # 002
        df['002'] = df['002'].replace('Много раз в день', 1)
        df['002'] = df['002'].replace('1-2 раза в день', 2)
        df['002'] = df['002'].replace('Несколько раз в неделю', 3)
        df['002'] = df['002'].replace('Раз в неделю', 4)
        df['002'] = df['002'].replace('Реже', 5)
        df['002'] = self.__normalize_scale_1_to_5(df['002'])

        for i in range(7, 135, 1):
            col_name = '{:03d}'.format(i)
            df[col_name] = self.__normalize_scale_1_to_5(df[col_name])

        #
        # clusterization
        #

        X = TSNE(n_components=2).fit_transform(df)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
        clusterer.fit(X)

        labels = clusterer.labels_
        min_label = np.min(labels)
        for i in range(len(df)):
            labels[i] -= min_label

        n_colors = len(np.unique(clusterer.labels_))
        cmap = cm.get_cmap('jet')
        colors = []
        for i in range(n_colors):
            colors.append(cmap((i + 0.5) / n_colors))

        plt.figure(figsize=(10, 10))
        for i in range(X.shape[0]):
            plt.scatter(X[i, 0], X[i, 1], s=100, color=colors[labels[i]])
        plt.savefig('clusterization.png', bbox_inches='tight', dpi=200)
        plt.close()


if __name__ == '__main__':
    processor = Processor(path_to_xlsx='data_input.xlsx')
