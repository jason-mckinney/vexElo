import requests
import pandas as pd
from pathlib import Path
import numpy as np
import time

k_factor = 64
nToEstablish = 16
pd.options.mode.chained_assignment = None

def get_all_teams():
    teams_total = requests.get(
        'https://api.vexdb.io/v1/get_teams',
        params={
            'nodata': 'true',
            'program': 'VRC'
        }
    ).json()['size']

    res = requests.get(
        'https://api.vexdb.io/v1/get_teams',
        params={
          'program': 'VRC'
        }
    ).json()

    teams = res['result']

    while len(teams) < teams_total:
        res = requests.get(
            'https://api.vexdb.io/v1/get_teams',
            params={
                'program': 'VRC',
                'limit_start': len(teams)
            }
        ).json()
        teams.extend(res['result'])

    return pd.DataFrame(teams).set_index('number')


def get_all_matches(season='current'):
    matches_total = requests.get(
        'https://api.vexdb.io/v1/get_matches',
        params={'nodata': 'true', 'season': season}
    ).json()['size']

    res = requests.get(
        'https://api.vexdb.io/v1/get_matches',
        params={
            'season': season
        }
    ).json()

    matches = res['result']

    while len(matches) < matches_total:
        res = requests.get(
            'https://api.vexdb.io/v1/get_matches',
            params={
                'limit_start': len(matches),
                'season': season
            }
        ).json()
        matches.extend(res['result'])
    raw_dataframe = pd.DataFrame(matches)
    raw_dataframe = raw_dataframe[raw_dataframe.blue2 != '']
    raw_dataframe = raw_dataframe[raw_dataframe.scored == 1]

    skus = raw_dataframe['sku'].drop_duplicates()[::-1]
    dataframe = pd.DataFrame(columns=[
            'blue1',
            'blue2',
            'blue3',
            'bluescore',
            'bluesit',
            'division',
            'field',
            'instance',
            'matchnum',
            'red1',
            'red2',
            'red3',
            'redscore',
            'redsit',
            'round',
            'scheduled',
            'scored',
            'sku'
    ])

    for sku in skus.values:
        dataframe = dataframe.append(raw_dataframe[raw_dataframe['sku'] == sku])

    return dataframe


def elo_rankings_from_matches(team_list, matches, rankings=None):
    def add_team(the_list, team):
        try:
            team_from_list = team_list.loc[team]
            country = team_from_list[1]
            region = team_from_list[6]
            grade = team_from_list[2]
        except KeyError:
            region = ''
            grade = ''
            country = ''

        mean_elo = the_list[the_list[:, 5] == False][:, 2].mean()
        return np.insert(the_list, 0, [team, 0, mean_elo, 0, 0, True, 0.0, region, country, grade], axis=0)


    def award_match(match, ranks):
        # 0,     1,     2,         3,    4,    5
        # blue1, blue2, bluescore, red1, red2, redscore

        if match[0] not in ranks[:, 0]:
            ranks = add_team(ranks, match[0])
        if match[1] not in ranks[:, 0]:
            ranks = add_team(ranks, match[1])
        if match[3] not in ranks[:, 0]:
            ranks = add_team(ranks, match[3])
        if match[4] not in ranks[:, 0]:
            ranks = add_team(ranks, match[4])

        blue1 = np.where(ranks[:, 0] == match[0])[0][0]
        blue2 = np.where(ranks[:, 0] == match[1])[0][0]
        red1 = np.where(ranks[:, 0] == match[3])[0][0]
        red2 = np.where(ranks[:, 0] == match[4])[0][0]

        blue_r1 = ranks[blue1, 2]
        blue_r2 = ranks[blue2, 2]
        red_r1 = ranks[red1, 2]
        red_r2 = ranks[red2, 2]

        blue_rating = (blue_r1 + blue_r2) / 2.0
        red_rating = (red_r1 + red_r2) / 2.0
        expected_blue = 1.0 / (1.0 + pow(10.0, ((red_rating - blue_rating) / 400.0)))
        expected_red = 1.0 - expected_blue

        if match[2] > match[5]:
            actual_blue = 1.0
            ranks[blue1, 4] += 1
            ranks[blue2, 4] += 1
        elif match[2] < match[5]:
            actual_blue = 0.0
            ranks[red1, 4] += 1
            ranks[red2, 4] += 1
        else:
            actual_blue = 0.5

        actual_red = 1.0 - actual_blue

        delta_blue = k_factor * (actual_blue - expected_blue)
        delta_red = k_factor * (actual_red - expected_red)
        blue1_contrib = blue_r1 / (blue_rating * 2)
        blue2_contrib = 1.0 - blue1_contrib
        red1_contrib = red_r1 / (red_rating * 2)
        red2_contrib = 1.0 - red1_contrib

        if ranks[blue1, 5]:
            modifier = 0

            if actual_blue == 1.0:
                modifier = 400 - int(ranks[red1, 5]) * 100 - int(ranks[red2, 5]) * 100
            elif actual_blue == 0.0:
                modifier = -400 + int(ranks[red1, 5]) * 100 + int(ranks[red2, 5]) * 100

            ranks[blue1, 6] += red_rating + modifier
        else:
            ranks[blue1, 2] = max(100.0, ranks[blue1, 2] + delta_blue * blue1_contrib)

        if ranks[blue2, 5]:
            modifier = 0

            if actual_blue == 1.0:
                modifier = 400 - int(ranks[red1, 5]) * 100 - int(ranks[red2, 5]) * 100
            elif actual_blue == 0.0:
                modifier = -400 + int(ranks[red1, 5]) * 100 + int(ranks[red2, 5]) * 100

            ranks[blue2, 6] += red_rating + modifier
        else:
            ranks[blue2, 2] = max(100.0, ranks[blue2, 2] + delta_blue * blue2_contrib)

        if ranks[red1, 5]:
            modifier = 0

            if actual_red == 1.0:
                modifier = 400 - int(ranks[blue1, 5]) * 100 - int(ranks[blue2, 5]) * 100
            elif actual_red == 0.0:
                modifier = -400 + int(ranks[blue2, 5]) * 100 + int(ranks[blue2, 5]) * 100

            ranks[red1, 6] += blue_rating + modifier
        else:
            ranks[red1, 2] = max(100.0, ranks[red1, 2] + delta_red * red1_contrib)

        if ranks[red2, 5]:
            modifier = 0

            if actual_red == 1.0:
                modifier = 400 - int(ranks[blue1, 5]) * 100 - int(ranks[blue2, 5]) * 100
            elif actual_red == 0.0:
                modifier = -400 + int(ranks[blue2, 5]) * 100 + int(ranks[blue2, 5]) * 100

            ranks[red2, 6] += blue_rating + modifier
        else:
            ranks[red2, 2] = max(100.0, ranks[red2, 2] + delta_red * red2_contrib)

        ranks[blue1, 3] += 1
        ranks[blue2, 3] += 1
        ranks[red1, 3] += 1
        ranks[red2, 3] += 1

        if ranks[blue1, 5]:
            ranks[blue1, 2] = max(100.0, ranks[blue1, 6] / ranks[blue1, 3])

            if ranks[blue1, 3] >= nToEstablish:
                ranks[blue1, 5] = False

        if ranks[blue2, 5]:
            ranks[blue2, 2] = max(100.0, ranks[blue2, 6] / ranks[blue2, 3])

            if ranks[blue2, 3] >= nToEstablish:
                ranks[blue2, 5] = False

        if ranks[red1, 5]:
            ranks[red1, 2] = max(100.0, ranks[red1, 6] / ranks[red1, 3])

            if ranks[red1, 3] >= nToEstablish:
                ranks[red1, 5] = False

        if ranks[red2, 5]:
            ranks[red2, 2] = max(100.0, ranks[red2, 6] / ranks[red2, 3])

            if ranks[red2, 3] >= nToEstablish:
                ranks[red2, 5] = False

        return ranks

    if rankings is None:
        rankings = pd.DataFrame(
            data={
                'global rank': [0],
                'team': ['0000'],
                'elo': [800.0],
                'played': [1],
                'won': [1],
                'provisional': [False],
                'provision': [800.0],
                'region': [''],
                'country': [''],
                'grade': ['']
            },
            columns=[
                'global rank',
                'team',
                'elo',
                'played',
                'won',
                'provisional',
                'provision',
                'region',
                'country',
                'grade'
            ]
        ).set_index('team')

    matches = matches.filter(
        items=[
            'blue1',
            'blue2',
            'bluescore',
            'red1',
            'red2',
            'redscore'
        ]
    )

    np_rankings = rankings.reset_index().to_numpy()

    for row in matches.values:
        np_rankings = award_match(row, np_rankings)

    rankings = pd.DataFrame(
        np_rankings,
        columns=[
            'team', 'global rank', 'elo', 'played', 'won',
            'provisional', 'provision', 'region', 'country', 'grade'
        ]
    ).set_index('team').drop('0000').reset_index().set_index('global rank')

    rankings.sort_values(by=['elo'], ascending=False, inplace=True)
    rankings.index = range(1, len(rankings) + 1)
    rankings = rankings.reindex(
        columns=['team', 'elo', 'played', 'won', 'region', 'country', 'grade', 'provisional', 'provision']
    )

    return rankings


if __name__ == '__main__':
    teams = get_all_teams()
    matchList = get_all_matches()
    elo_rankings_from_matches(teams, matchList)
