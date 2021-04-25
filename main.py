import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import split
from shapely.affinity import translate
from collections import Counter
from os.path import exists
from os import remove
from numpy import mean, std, sqrt, nan

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage[utf8]{inputenc}')
rc('text.latex', preamble=r'\usepackage[russian]{babel}')


class Processor:
    def __init__(self, **kwargs):
        self.__path_to_xlsx = kwargs['path_to_xlsx']

        self.__path_to_all = 'all.txt'
        self.__path_to_z, self.__path_to_y, self.__path_to_x = 'Z.txt', 'Y.txt', 'X.txt'

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

        # all
        df_statements1, df_statements2, df_statements, df_belief, df_social = self.__split_by_questions(df)
        self.__calculate_core(df_statements, label='Утверждения', path_to_result=self.__path_to_all, positive_coeff=True)
        self.__calculate_core(df_statements, label='Утверждения', path_to_result=self.__path_to_all, positive_coeff=False)
        self.__calculate_core(df_statements1, label='Представления', path_to_result=self.__path_to_all, positive_coeff=True)
        self.__calculate_core(df_statements1, label='Представления', path_to_result=self.__path_to_all, positive_coeff=False)
        self.__calculate_core(df_statements2, label='Критерии', path_to_result=self.__path_to_all, positive_coeff=True)
        self.__calculate_core(df_statements2, label='Критерии', path_to_result=self.__path_to_all, positive_coeff=False)

        # z
        df_z_statements1, df_z_statements2, df_z_statements, df_z_belief, df_z_social = self.__split_by_questions(df_z)
        self.__calculate_core(df_z_statements, label='Утверждения', path_to_result=self.__path_to_z, positive_coeff=True)
        self.__calculate_core(df_z_statements, label='Утверждения', path_to_result=self.__path_to_z, positive_coeff=False)
        self.__calculate_core(df_z_statements1, label='Представления', path_to_result=self.__path_to_z, positive_coeff=True)
        self.__calculate_core(df_z_statements1, label='Представления', path_to_result=self.__path_to_z, positive_coeff=False)
        self.__calculate_core(df_z_statements2, label='Критерии', path_to_result=self.__path_to_z, positive_coeff=True)
        self.__calculate_core(df_z_statements2, label='Критерии', path_to_result=self.__path_to_z, positive_coeff=False)

        # y
        df_y_statements1, df_y_statements2, df_y_statements, df_y_belief, df_y_social = self.__split_by_questions(df_y)
        self.__calculate_core(df_y_statements, label='Утверждения', path_to_result=self.__path_to_y, positive_coeff=True)
        self.__calculate_core(df_y_statements, label='Утверждения', path_to_result=self.__path_to_y, positive_coeff=False)
        self.__calculate_core(df_y_statements1, label='Представления', path_to_result=self.__path_to_y, positive_coeff=True)
        self.__calculate_core(df_y_statements1, label='Представления', path_to_result=self.__path_to_y, positive_coeff=False)
        self.__calculate_core(df_y_statements2, label='Критерии', path_to_result=self.__path_to_y, positive_coeff=True)
        self.__calculate_core(df_y_statements2, label='Критерии', path_to_result=self.__path_to_y, positive_coeff=False)

        # x
        df_x_statements1, df_x_statements2, df_x_statements, df_x_belief, df_x_social = self.__split_by_questions(df_x)
        self.__calculate_core(df_x_statements, label='Утверждения', path_to_result=self.__path_to_x, positive_coeff=True)
        self.__calculate_core(df_x_statements, label='Утверждения', path_to_result=self.__path_to_x, positive_coeff=False)
        self.__calculate_core(df_x_statements1, label='Представления', path_to_result=self.__path_to_x, positive_coeff=True)
        self.__calculate_core(df_x_statements1, label='Представления', path_to_result=self.__path_to_x, positive_coeff=False)
        self.__calculate_core(df_x_statements2, label='Критерии', path_to_result=self.__path_to_x, positive_coeff=True)
        self.__calculate_core(df_x_statements2, label='Критерии', path_to_result=self.__path_to_x, positive_coeff=False)

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
            # print(feature)
            col = df[feature]
            # print(col.head(25))
            for i in range(len(col)):
                if col[i] is nan or col[i] is None or col[i] == '':
                    # to_drop.append((i, feature))
                    raise Exception('Empty cell in ({:03d}, {})'.format(i, feature))
        # print(to_drop)

    def __readout(self):
        columns = self.__base_columns + ['{:03d}'.format(i + 1) for i in range(self.__n_questions)]
        df = pd.read_excel(self.__path_to_xlsx)

        len_base = len(self.__base_columns)
        with open('features.txt', 'w') as f:
            for i in range(len_base, len(columns), 1):
                f.write('{}\n{}\n\n'.format(columns[i], df.columns[i]))

        df.columns = columns

        return df

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

        with open(self.__path_to_all, 'a') as f:
            f.write('Число респондентов: {:03d}\n'.format(n))
            f.write('\n######### ПОЛ #########\n')
            f.write('Мужской: {:03d} ({:04.2f}%)\n'.format(n_male, n_male / n * 100))
            f.write('Женский: {:03d} ({:04.2f}%)\n'.format(n_female, n_female / n * 100))
            f.write('\n####### ВОЗРАСТ #######\n')
            f.write('Z: {:03d} ({:04.2f}%)\n'.format(n_z, n_z / n * 100))
            f.write('Y: {:03d} ({:04.2f}%)\n'.format(n_y, n_y / n * 100))
            f.write('X: {:03d} ({:04.2f}%)\n'.format(n_x, n_x / n * 100))

        with open(self.__path_to_z, 'a') as f:
            f.write('Число респондентов: {:03d}\n'.format(n_z))
            f.write('\n######### ПОЛ #########\n')
            f.write('Мужской пол: {:03d} ({:04.2f}%)\n'.format(n_z_male, n_z_male / n_z * 100))
            f.write('Женский пол: {:03d} ({:04.2f}%)\n'.format(n_z_female, n_z_female / n_z * 100))

        with open(self.__path_to_y, 'a') as f:
            f.write('Число респондентов: {:03d}\n'.format(n_y))
            f.write('\n######### ПОЛ #########\n')
            f.write('Мужской пол: {:03d} ({:04.2f}%)\n'.format(n_y_male, n_y_male / n_y * 100))
            f.write('Женский пол: {:03d} ({:04.2f}%)\n'.format(n_y_female, n_y_female / n_y * 100))

        with open(self.__path_to_x, 'a') as f:
            f.write('Число респондентов: {:03d}\n'.format(n_x))
            f.write('\n######### ПОЛ #########\n')
            f.write('Мужской пол: {:03d} ({:04.2f}%)\n'.format(n_x_male, n_x_male / n_x * 100))
            f.write('Женский пол: {:03d} ({:04.2f}%)\n'.format(n_x_female, n_x_female / n_x * 100))

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
                        'страна Казахстан, город Атырау', 'Бостон, США'):
                city = 'Москва'
            elif city in ('Одинцово', 'Люберцы', 'Ступино', 'Ступино московской обл',
                          'Мытищи', 'Химки', 'Балашиха', 'Ступино Московской области',
                          'Г. Троицк, Москва', 'Хотьково', 'Пушкино', 'Железнодорожный',
                          'Ступино Московской обл', 'Серпухов', 'Котельники', 'Сергиев Посад'):
                city = 'Московская область'
            elif city in ('Волгодонск', 'Волгодоск', 'Ростовская обл', 'Живу в деревне', 'Азов', 'Волгодонск Ростовская обл.',
                          'Ростов-на-Дону'):
                city = 'Ростовская область'
            elif city in ('питер', 'Ереван', 'Санкт петеобург', 'Санкт-Перетбург', 'СПб', 'Вентспилс', 'Волхов'):
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
        # first_qq = ['{:03d}'.format(i) for i in range(1, 7, 1)]
        statements1 = ['{:03d}'.format(i) for i in range(self.__n_statements1_min, self.__n_statements1_max + 1, 1)]
        statements2 = ['{:03d}'.format(i) for i in range(self.__n_statements2_min, self.__n_statements2_max + 1, 1)]
        belief = ['{:03d}'.format(i) for i in range(self.__n_belief_min, self.__n_belief_max + 1, 1)]
        social = ['{:03d}'.format(i) for i in range(self.__n_social_min, self.__n_social_max + 1, 1)]

        df_statements1, df_statements2, df_statements, df_belief, df_social = \
            df[statements1], df[statements2], df[statements1 + statements2], df[belief], df[social]

        return df_statements1, df_statements2, df_statements, df_belief, df_social

    @staticmethod
    def __calculate_core(df, label, path_to_result, positive_coeff=True):
        if positive_coeff:
            target_answers = ('Скорее согласен', 'Полностью согласен')
        else:
            target_answers = ('Скорее не согласен', 'Полностью не согласен')

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

        with open(path_to_result, 'a') as f:
            if positive_coeff:
                f.write('\n######### КПО: {} #########\n'.format(label))
            else:
                f.write('\n######### КНО: {} #########\n'.format(label))
            f.write('Число утверждений: {:03d}\n'.format(n_features))
            for i in range(n_features):
                f.write('{}: {:04.1f}\n'.format(features[i], coeff_pos[i]))
            f.write('Математическое ожидание M: {:04.1f}\n'.format(mean_coeff))
            f.write('Стандартное отклонение s: {:04.1f}\n'.format(sigma_coeff))
            f.write('Граница M + s: {:04.1f}\n'.format(mean_coeff + sigma_coeff))
            f.write('ЯДРО:\n')
            f.write('Число утверждений: {:03d} ({:04.1f}%)\n'.format(len(core), len(core) / n_features * 100))
            for i in range(len(core)):
                f.write('{}: {:03.1f}\n'.format(core[i][0], core[i][1]))
            f.write('ПЕРИФЕРИЯ, БЛИЗКАЯ К ЯДРУ:\n')
            f.write('Число утверждений: {:03d} ({:04.1f}%)\n'.format(len(precore), len(precore) / n_features * 100))
            for i in range(len(precore)):
                f.write('{}: {:04.1f}\n'.format(precore[i][0], precore[i][1]))
            f.write('ПЕРИФЕРИЯ:\n')
            f.write('Число утверждений: {:03d} ({:04.1f}%)\n'.format(len(periphery), len(periphery) / n_features * 100))
            for i in range(len(periphery)):
                f.write('{}: {:04.1f}\n'.format(periphery[i][0], periphery[i][1]))


if __name__ == '__main__':
    processor = Processor(path_to_xlsx='data.xlsx')
