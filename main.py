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

        cities_statistics = self.__get_cities_statistics(df_initial)
        self.__plot_cities_on_map()

    def __get_cities_statistics(self, df):

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

        # select only top
        n_top = 5
        counts = Counter(df['city'])
        cc = [(k, v) for k, v in counts.items()]
        cc.sort(key=lambda x: x[1], reverse=True)
        cc_agg = cc[:n_top]
        n_other = 0
        for i in range(n_top, len(cc)):
            n_other += cc[i][1]
        cc_agg.append(('Другие города', n_other))
        counts_agg = dict(cc_agg)

        # make dataframe
        df_cities = pd.DataFrame.from_dict(counts_agg, orient='index', columns=['number'])
        df_cities['percent'] = df_cities['number'] / len(df) * 100
        df_cities = df_cities.sort_values(by=['number'])

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

    def __readout(self):
        n_questions = 134
        base_columns = ['time', 'age', 'sex', 'education_level', 'education_profile', 'city']
        columns = base_columns + ['{:03d}'.format(i + 1) for i in range(n_questions)]
        df = pd.read_excel(self.__path_to_xlsx,
                           names=columns)

        return df

    def __plot_cities_on_map(self):
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
        # # df = pd.DataFrame(
        #     {'City': ['Buenos Aires', 'Brasilia', 'Santiago', 'Bogota', 'Caracas'],
        #      'Country': ['Argentina', 'Brazil', 'Chile', 'Colombia', 'Venezuela'],
        #      'Latitude': [-34.58, -15.78, -33.45, 4.60, 10.48],
        #      'Longitude': [-58.66, -47.91, -70.66, -74.08, -66.86]})

        # gdf = geopandas.GeoDataFrame(
        #     df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude))

        # print(gdf.head())

        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        russia = world[world['iso_a3'] == 'RUS']
        new_rus = shift_geom(-90, shift_geom(90, russia, False), True)
        new_rus.plot(figsize=(20, 20))
        plt.scatter([75, 100], [60, 70], c='red', s=100, alpha=1)
        plt.text(75, 60, 'Москва', color='black', fontsize=20, zorder=1)
        plt.xticks([])
        plt.yticks([])
        plt.savefig('russia.png', bbox_inches='tight', dpi=200)

        # states = gpd.read_file('data/usa-states-census-2014.shp')
        # states = states.to_crs("EPSG:3395")
        # states.boundary.plot()


if __name__ == '__main__':
    processor = Processor(path_to_xlsx='data.xlsx')
