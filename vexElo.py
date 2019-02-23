import requests
import json
import pandas as pd
import numpy as np
import asyncio
import re
from pathlib import Path

def get_all_matches():
    matches_total = requests.get(
        'https://api.vexdb.io/v1/get_matches',
        params={'nodata': 'true', 'season': 'current'}
    ).json()['size']

    res = requests.get(
        'https://api.vexdb.io/v1/get_matches',
        params={
            'season': 'current'
        }
    ).json()

    matches = res['result']

    while len(matches) < matches_total:
        res = requests.get(
            'https://api.vexdb.io/v1/get_matches',
            params={
                'limit_start': len(matches),
                'season': 'current'
            }
        ).json()
        matches.extend(res['result'])
    dataframe = pd.DataFrame(matches)
    dataframe = dataframe[dataframe.blue2 != '']
    dataframe = dataframe[dataframe.scored == 1]
    dataframe.to_csv(Path('.') / 'eloVRC.csv')


def elo_rankings_from_file():
    dataframe = pd.read_csv(Path('.') / 'eloVRC.csv').sort_index(ascending = False).filter(
        items=[
            'blue1',
            'blue2',
            'bluescore',
            'red1',
            'red2',
            'redscore',
        ]
    )

    for row in dataframe.iterrows():




if __name__ == '__main__':
    elo_rankings_from_file()
