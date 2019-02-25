import requests
import pandas as pd
from pathlib import Path

kfactor = 64
nToEstablish = 16

def get_all_matches(season = 'current'):
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


def elo_rankings_from_file():
    def add_team(df, team):
        return df.append(pd.DataFrame(
            data={
                'team': [team],
                'elo': [df.loc[df['provisional']]['elo'].mean()],
                'played': [0],
                'won': [0],
                'provisional': [True],
                'provision': [0.0]
            }
        ).set_index('team', inplace=False))

    def award_match(match, eloDB):
        if match[1].blue1 not in eloDB.index:
            eloDB = add_team(eloDB, match[1].blue1)
        if match[1].blue2 not in eloDB.index:
            eloDB = add_team(eloDB, match[1].blue2)
        if match[1].red1 not in eloDB.index:
            eloDB = add_team(eloDB, match[1].red1)
        if match[1].red2 not in eloDB.index:
            eloDB = add_team(eloDB, match[1].red2)

        blue1 = eloDB.loc[match[1].blue1]
        blue2 = eloDB.loc[match[1].blue2]
        red1 = eloDB.loc[match[1].red1]
        red2 = eloDB.loc[match[1].red2]

        blueR1 = blue1['elo']
        blueR2 = blue2['elo']
        redR1 = red1['elo']
        redR2 = red2['elo']

        blueRating = (blueR1 + blueR2)/2.0
        redRating = (redR1 + redR2)/2.0
        expectedBlue = 1.0/(1.0 + pow(10.0, ((redRating - blueRating) / 400.0)))
        expectedRed = 1.0 - expectedBlue
        actualBlue = 0.0

        if match[1].bluescore > match[1].redscore:
            actualBlue = 1.0
            eloDB.at[blue1.name, 'won'] += 1
            eloDB.at[blue2.name, 'won'] += 1
        elif match[1].bluescore < match[1].redscore:
            actualBlue = 0.0
            eloDB.at[red1.name, 'won'] += 1
            eloDB.at[red2.name, 'won'] += 1
        else:
            actualBlue = 0.5

        actualRed = 1.0 - actualBlue

        deltaBlue = kfactor * (actualBlue - expectedBlue)
        deltaRed = kfactor * (actualRed - expectedRed)
        blue1Contrib = blueR1 / blueRating
        blue2Contrib = 1.0 - blue1Contrib
        red1Contrib = redR1 / redRating
        red2Contrib = 1.0 - red1Contrib

        if blue1.provisional:
            modifier = 0

            if actualBlue is 1.0:
                modifier = 400 - int(red1.provisional) * 100 - int(red2.provisional) * 100
            elif actualBlue is 0:
                modifier = -400 + int(red1.provisional) * 100 + int(red2.privisional) * 100

            eloDB.at[blue1.name, 'provision'] += redRating + modifier
        else:
            eloDB.at[blue1.name, 'elo'] = max(blue1['elo'] + kfactor * deltaBlue * blue1Contrib, 100)

        if blue2.provisional:
            modifier = 0

            if actualBlue is 1.0:
                modifier = 400 - int(red1.provisional) * 100 - int(red2.provisional) * 100
            elif actualBlue is 0:
                modifier = -400 + int(red1.provisional) * 100 + int(red2.privisional) * 100

            eloDB.at[blue2.name, 'provision'] += redRating + modifier
        else:
            eloDB.at[blue2.name, 'elo'] = max(blue2['elo'] + kfactor * deltaBlue * blue2Contrib, 100)

        if red1.provisional:
            modifier = 0

            if actualRed is 1.0:
                modifier = 400 - int(blue1.provisional) * 100 - int(blue2.provisional) * 100
            elif actualRed is 0:
                modifier = -400 + int(blue1.provisional) * 100 + int(blue2.privisional) * 100

            eloDB.at[red1.name, 'provision'] += blueRating + modifier
        else:
            eloDB.at[red1.name, 'elo'] = max(red1['elo'] + kfactor * deltaRed * red1Contrib, 100)

        if red2.provisional:
            modifier = 0

            if actualRed is 1.0:
                modifier = 400 - int(blue1.provisional) * 100 - int(blue2.provisional) * 100
            elif actualRed is 0:
                modifier = -400 + int(blue1.provisional) * 100 + int(blue2.privisional) * 100

            eloDB.at[red2.name, 'provision'] += blueRating + modifier
        else:
            eloDB.at[red2.name, 'elo'] = max(red2['elo'] + kfactor * deltaRed * red2Contrib, 100)

        eloDB.at[blue1.name, 'played'] += 1
        eloDB.at[blue2.name, 'played'] += 1
        eloDB.at[red1.name, 'played'] += 1
        eloDB.at[red2.name, 'played'] += 1

        if eloDB.loc[blue1.name, 'provisional'] and eloDB.loc[blue1.name, 'played'] >= nToEstablish:
            eloDB.at[blue1.name, 'provisional'] = False
            eloDB.at[blue1.name, 'elo'] = eloDB.loc[blue1.name, 'provision'] / eloDB.loc[blue1.name, 'played']

        if eloDB.loc[blue2.name, 'provisional'] and eloDB.loc[blue2.name, 'played'] >= nToEstablish:
            eloDB.at[blue2.name, 'provisional'] = False
            eloDB.at[blue2.name, 'elo'] = eloDB.loc[blue2.name, 'provision'] / eloDB.loc[blue2.name, 'played']

        if eloDB.loc[red1.name, 'provisional'] and eloDB.loc[red1.name, 'played'] >= nToEstablish:
            eloDB.at[red1.name, 'provisional'] = False
            eloDB.at[red1.name, 'elo'] = eloDB.loc[red1.name, 'provision'] / eloDB.loc[red1.name, 'played']

        if eloDB.loc[red2.name, 'provisional'] and eloDB.loc[red2.name, 'played'] >= nToEstablish:
            eloDB.at[red2.name, 'provisional'] = False
            eloDB.at[red2.name, 'elo'] = eloDB.loc[red2.name, 'provision'] / eloDB.loc[red2.name, 'played']

        return eloDB


    teamRow = {
        'team': [44],
        'elo': [800.0],
        'played': [9999],
        'won': [9999],
        'provisional': [True],
        'provision': [800.0]
    }

    teamDB = pd.DataFrame(data=teamRow).set_index('team', inplace=False)

    matches = pd.read_csv(Path('.') / 'matches.csv').sort_index(ascending = False).filter(
        items=[
            'blue1',
            'blue2',
            'bluescore',
            'red1',
            'red2',
            'redscore',
        ]
    )

    for row in matches.iterrows():
        teamDB = award_match(row, teamDB)
    print(teamDB)
    teamDB.to_csv(Path('.') / 'elo.csv')


if __name__ == '__main__':
    get_all_matches()
    elo_rankings_from_file()
