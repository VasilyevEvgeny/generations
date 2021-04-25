import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import split
from shapely.affinity import translate
from collections import Counter

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage[utf8]{inputenc}')
rc('text.latex', preamble=r'\usepackage[russian]{babel}')


class Processor:
    def __init__(self, **kwargs):
        self.__path_to_xlsx = kwargs['path_to_xlsx']

        df_initial = self.__readout()

        # cities
        df = self.__preprocess_cities(df_initial)
        df_cities, n_top = self.__process_cities(df)
        self.__plot_cities_on_map(df_cities, n_top)

    def __readout(self):
        n_questions = 134
        base_columns = ['time', 'age', 'sex', 'education_level', 'education_profile', 'city']
        columns = base_columns + ['{:03d}'.format(i + 1) for i in range(n_questions)]
        df = pd.read_excel(self.__path_to_xlsx,
                           names=columns)

        return df

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
            elif city in ('Волгодонск', 'Ростовская обл', 'Живу в деревне', 'Азов', 'Волгодонск Ростовская обл.',
                          'Ростов-на-Дону'):
                city = 'Ростовская область'
            elif city in ('питер', 'Ереван', 'Санкт петеобург', 'СПб', 'Вентспилс'):
                city = 'Санкт-Петербург'
            elif city in ('райцентр Курской области'):
                city = 'Курск'
            elif city in ('вологда', 'Череповец'):
                city = 'Вологда'
            elif city in ('Кишинев'):
                city = 'Екатеринбург'
            elif city in ('Березники(Пермский край)'):
                city = 'Пермь'
            elif city in ('Усолье-Сибирское (Иркутская область)'):
                city = 'Иркутск'
            elif city in ('Самарская область,с Челно-вершины'):
                city = 'Самара'
            elif city in ('ВОЛГОГРАД'):
                city = 'Волгоград'
            elif city in ('Суворов'):
                city = 'Тула'
            elif city in ('Ростов'):
                city = 'Ярославль'
            elif city in ('Железногорск'):
                city = 'Красноярск'
            elif city in ('Краснодар', 'Невинномысск', 'Севастополь', 'Бишкек', 'Пятигорск'):
                city = 'Краснодарский край'

            df.loc[i, 'city'] = city
        df = df.drop(df.index[i_to_drop])

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
                  'Новосибирск': (46.0154, 51.5462),
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


if __name__ == '__main__':
    processor = Processor(path_to_xlsx='data.xlsx')
