# -----------------------------------------------------------------------------
#                                  MIT License
#
#                        Copyright (c) 2020 Jason McKinney
#
# 	Permission is hereby granted, free of charge, to any person obtaining a copy
# 	of this software and associated documentation files (the "Software"), to deal
# 	in the Software without restriction, including without limitation the rights
# 	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# 	copies of the Software, and to permit persons to whom the Software is
# 	furnished to do so, subject to the following conditions:
#
# 	The above copyright notice and this permission notice shall be included in all
# 	copies or substantial portions of the Software.
#
# 	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# 	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# 	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# 	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# 	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# 	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# 	SOFTWARE.
#
# --------------------------------------------------------------------------------
# 	vecElo.py
#
# 	Created:  2019-02-22
# --------------------------------------------------------------------------------
# 	The author can be contacted via email at jason_at_jmmckinney_dot_net
# 	or on the VEX Forum as jmmckinney.
# --------------------------------------------------------------------------------


import requests
import pandas as pd
import numpy as np
import pickle
import os.path
import math
import numbers
import signal
from threading import Event

k_factor = 64
nToEstablish = 16
pd.options.mode.chained_assignment = None

exit_event = Event()


class Metadata:
    def __init__(self, scored_matches=0):
        self.scored_matches = scored_matches


def get_teams_total(program='VRC'):
    return requests.get(
        'https://api.vexdb.io/v1/get_teams',
        params={
            'nodata': 'true',
            'program': program
        }
    ).json()['size']


def get_all_teams(program='VRC'):
    teams_total = get_teams_total(program)

    res = requests.get(
        'https://api.vexdb.io/v1/get_teams',
        params={
          'program': program
        }
    ).json()

    result_teams = res['result']

    while len(result_teams) < teams_total:
        res = requests.get(
            'https://api.vexdb.io/v1/get_teams',
            params={
                'program': program,
                'limit_start': len(result_teams)
            }
        ).json()
        result_teams.extend(res['result'])

    return pd.DataFrame(result_teams).set_index('number')


def get_matches_total(season='current'):
    return requests.get(
        'https://api.vexdb.io/v1/get_matches',
        params={'nodata': 'true', 'season': season}
    ).json()['size']


def get_matches_scored(season='current'):
    return requests.get(
        'https://api.vexdb.io/v1/get_matches',
        params={'nodata': 'true', 'season': season, 'scored': '1'}
    ).json()['size']


def get_all_matches(season='current'):
    matches_total = get_matches_total(season)
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
        dataframe = dataframe.append(raw_dataframe[raw_dataframe['sku'] == sku], sort=False)

    return dataframe


def elo_rankings_from_matches(team_list, matches, rankings=None):
    def add_team(the_list, team):
        try:
            team_from_list = team_list.loc[team]
            country = team_from_list['country']
            region = team_from_list['region']
            grade = team_from_list['grade']
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
        np_rankings = rankings.reset_index().to_numpy()
    else:
        np_rankings = rankings.reset_index().to_numpy()
        np_rankings[:, 0:10] = np_rankings[:, [1, 0, 2, 3, 4, 8, 9, 5, 6, 7]]

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

    for row in matches.values:
        np_rankings = award_match(row, np_rankings)

    rankings = pd.DataFrame(
        np_rankings,
        columns=[
            'team', 'global rank', 'elo', 'played', 'won',
            'provisional', 'provision', 'region', 'country', 'grade'
        ]
    ).set_index('team')

    if '0000' in rankings.index:
        rankings.drop('0000', inplace=True)

    rankings = rankings.reset_index().set_index('global rank')
    rankings.sort_values(by=['elo'], ascending=False, inplace=True)
    rankings.index = range(1, len(rankings) + 1)
    rankings = rankings.reindex(
        columns=['team', 'elo', 'played', 'won', 'region', 'country', 'grade', 'provisional', 'provision']
    )

    return rankings


def update_rankings(selected_season='current'):
    os.makedirs("data/" + selected_season, exist_ok=True)

    if os.path.exists("data/" + selected_season + "/metadata.pickle"):
        with open("data/" + selected_season + "/metadata.pickle", "rb") as file:
            metadata = pickle.load(file)

        if metadata.scored_matches == get_matches_scored(selected_season):
            return None

    print("new data uploaded, updating ratings...")

    if os.path.exists("data/" + selected_season + "/teams.pickle"):
        num_teams = get_teams_total()

        with open("data/" + selected_season + "/teams.pickle", "rb") as file:
            teams = pickle.load(file)

        if teams.shape[0] < num_teams:
            teams = get_all_teams()
            with open("data/" + selected_season + "/teams.pickle", "wb") as file:
                pickle.dump(teams, file)
    else:
        teams = get_all_teams()
        with open("data/" + selected_season + "/teams.pickle", "wb") as file:
            pickle.dump(teams, file)

    if os.path.exists("data/" + selected_season + "/match_list.pickle"):
        match_list = get_all_matches(selected_season)
        metadata = Metadata(get_matches_scored(selected_season))

        with open("data/" + selected_season + "/metadata.pickle", "wb") as file:
            pickle.dump(metadata, file)
        with open("data/" + selected_season + "/match_list.pickle", "rb") as file:
            match_list_old = pickle.load(file)
        with open("data/" + selected_season + "/match_list.pickle", "wb") as file:
            pickle.dump(match_list, file)

        match_list = match_list_old.merge(match_list, indicator=True, how='outer')
        match_list = match_list[match_list['_merge'] == 'right_only']
    else:
        match_list = get_all_matches(selected_season)
        metadata = Metadata(get_matches_scored(selected_season))

        with open("data/" + selected_season + "/metadata.pickle", "wb") as file:
            pickle.dump(metadata, file)
        with open("data/" + selected_season + "/match_list.pickle", "wb") as file:
            pickle.dump(match_list, file)

    if os.path.exists("data/" + selected_season + "/elo_db.pickle"):
        with open("data/" + selected_season + "/elo_db.pickle", "rb") as file:
            elo_db = pickle.load(file)
        elo_db = elo_rankings_from_matches(teams, match_list, elo_db)
    else:
        elo_db = elo_rankings_from_matches(teams, match_list)

    with open("data/" + selected_season + "/elo_db.pickle", "wb") as file:
        pickle.dump(elo_db, file)
    elo_db.to_csv("data/" + selected_season + "/elo_db.csv")
    return elo_db


def set_exit_signal(signo, _frame):
    global exit_event
    exit_event.set()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, set_exit_signal)

    while not exit_event.is_set():
        rankings = update_rankings('current')
        exit_event.wait(150.0)
