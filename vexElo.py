import requests
import pandas as pd
from pathlib import Path
import numpy as np

k_factor = 64
nToEstablish = 16


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
    dataframe = pd.DataFrame(matches)
    dataframe = dataframe[dataframe.blue2 != '']
    dataframe = dataframe[dataframe.scored == 1]
    dataframe.to_csv(Path('.') / 'matches.csv')


def elo_rankings_from_file(name='elo.csv'):
    def add_team(teams, team):
        return np.insert(teams, 0, [team, teams[teams[:, 4] == False][:, 1].mean(), 0, 0, True, 0.0], axis=0)

    def award_match(match, elo_db):
        # 0,     1,     2,         3,    4,    5
        # blue1, blue2, bluescore, red1, red2, redscore

        if match[0] not in elo_db[:, 0]:
            elo_db = add_team(elo_db, match[0])
        if match[1] not in elo_db[:, 0]:
            elo_db = add_team(elo_db, match[1])
        if match[3] not in elo_db[:, 0]:
            elo_db = add_team(elo_db, match[3])
        if match[4] not in elo_db[:, 0]:
            elo_db = add_team(elo_db, match[4])

        b1_index = np.where(elo_db[:, 0] == match[0])[0][0]
        b2_index = np.where(elo_db[:, 0] == match[1])[0][0]
        r1_index = np.where(elo_db[:, 0] == match[3])[0][0]
        r2_index = np.where(elo_db[:, 0] == match[4])[0][0]

        blue1 = elo_db[b1_index]
        blue2 = elo_db[b2_index]
        red1 = elo_db[r1_index]
        red2 = elo_db[r2_index]

        blue_r1 = blue1[1]
        blue_r2 = blue2[1]
        red_r1 = red1[1]
        red_r2 = red2[1]

        blue_rating = (blue_r1 + blue_r2) / 2.0
        red_rating = (red_r1 + red_r2) / 2.0
        expected_blue = 1.0 / (1.0 + pow(10.0, ((red_rating - blue_rating) / 400.0)))
        expected_red = 1.0 - expected_blue

        if match[2] > match[5]:
            actual_blue = 1.0
            elo_db[b1_index, 3] += 1
            elo_db[b2_index, 3] += 1
        elif match[2] < match[5]:
            actual_blue = 0.0
            elo_db[r1_index, 3] += 1
            elo_db[r2_index, 3] += 1
        else:
            actual_blue = 0.5

        actual_red = 1.0 - actual_blue

        delta_blue = k_factor * (actual_blue - expected_blue)
        delta_red = k_factor * (actual_red - expected_red)
        blue1_contrib = blue_r1 / (blue_rating * 2)
        blue2_contrib = 1.0 - blue1_contrib
        red1_contrib = red_r1 / (red_rating * 2)
        red2_contrib = 1.0 - red1_contrib

        if blue1[4]:
            modifier = 0

            if actual_blue == 1.0:
                modifier = 400 - int(red1[4]) * 100 - int(red2[4]) * 100
            elif actual_blue == 0.0:
                modifier = -400 + int(red1[4]) * 100 + int(red2[4]) * 100

            elo_db[b1_index, 5] += red_rating + modifier
        else:
            elo_db[b1_index, 1] = max(100.0, blue1[1] + delta_blue * blue1_contrib)

        if blue2[4]:
            modifier = 0

            if actual_blue == 1.0:
                modifier = 400 - int(red1[4]) * 100 - int(red2[4]) * 100
            elif actual_blue == 0.0:
                modifier = -400 + int(red1[4]) * 100 + int(red2[4]) * 100

            elo_db[b2_index, 5] += red_rating + modifier
        else:
            elo_db[b2_index, 1] = max(100.0, blue2[1] + delta_blue * blue2_contrib)

        if red1[4]:
            modifier = 0

            if actual_red == 1.0:
                modifier = 400 - int(blue1[4]) * 100 - int(blue2[4]) * 100
            elif actual_red == 0.0:
                modifier = -400 + int(blue2[4]) * 100 + int(blue2[4]) * 100

            elo_db[r1_index, 5] += blue_rating + modifier
        else:
            elo_db[r1_index, 1] = max(100.0, red1[1] + delta_red * red1_contrib)

        if red2[4]:
            modifier = 0

            if actual_red == 1.0:
                modifier = 400 - int(blue1[4]) * 100 - int(blue2[4]) * 100
            elif actual_red == 0.0:
                modifier = -400 + int(blue2[4]) * 100 + int(blue2[4]) * 100

            elo_db[r2_index, 5] += blue_rating + modifier
        else:
            elo_db[r2_index, 1] = max(100.0, red2[1] + delta_red * red2_contrib)

        elo_db[b1_index, 2] += 1
        elo_db[b2_index, 2] += 1
        elo_db[r1_index, 2] += 1
        elo_db[r2_index, 2] += 1

        if elo_db[b1_index, 4]:
            elo_db[b1_index, 1] = max(100.0, elo_db[b1_index, 5] / elo_db[b1_index, 2])

            if elo_db[b1_index, 2] >= nToEstablish:
                elo_db[b1_index, 4] = False

        if elo_db[b2_index, 4]:
            elo_db[b2_index, 1] = max(100.0, elo_db[b2_index, 5] / elo_db[b2_index, 2])

            if elo_db[b2_index, 2] >= nToEstablish:
                elo_db[b2_index, 4] = False

        if elo_db[r1_index, 4]:
            elo_db[r1_index, 1] = max(100.0, elo_db[r1_index, 5] / elo_db[r1_index, 2])

            if elo_db[r1_index, 2] >= nToEstablish:
                elo_db[r1_index, 4] = False

        if elo_db[r2_index, 4]:
            elo_db[r2_index, 1] = max(100.0, elo_db[r2_index, 5] / elo_db[r2_index, 2])

            if elo_db[r2_index, 2] >= nToEstablish:
                elo_db[r2_index, 4] = False

        return elo_db

    team_row = {
        'team': ['0000'],
        'elo': [1200.0],
        'played': [1],
        'won': [1],
        'provisional': [False],
        'provision': [1000.0]
    }

    team_db = pd.DataFrame(data=team_row).to_numpy()

    matches = pd.read_csv(Path('.') / 'matches.csv').sort_index(ascending=False).filter(
        items=[
            'blue1',
            'blue2',
            'bluescore',
            'red1',
            'red2',
            'redscore',
        ]
    )

    for row in matches.values:
        team_db = award_match(row, team_db)

    data_frame = pd.DataFrame(
        team_db,
        columns=['team', 'elo', 'played', 'won', 'provisional', 'provision']
    ).set_index('team').drop('0000')

    data_frame.sort_values(by=['elo'], ascending=False, inplace=True)
    data_frame.reset_index(inplace=True)
    data_frame.index = range(1, len(data_frame) + 1)
    data_frame.index.names = ['rank']
    data_frame.to_csv(Path('.') / name)


if __name__ == '__main__':
    get_all_matches('In The Zone')
    #elo_rankings_from_file('elo2018.csv')
