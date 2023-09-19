import requests
import pandas as pd
import os
import json

base_url = 'https://fantasy.premierleague.com/api/'
outdir = '/Users/nick/PycharmProjects/FPL_project/data'

# get general information
summary = requests.get(base_url+'bootstrap-static/').json()
players = summary['elements']
teams = summary['teams']

# get fixture data
fixtures = requests.get(base_url+ 'fixtures/').json()

def FPL_pull():
    base_url = 'https://fantasy.premierleague.com/api/'
    summary = requests.get(base_url + 'bootstrap-static/').json()
    players = summary['elements']
    teams = summary['teams']
    fixtures = requests.get(base_url + 'fixtures/').json()

    return teams, players, fixtures

# fetch player data - I think this is deprecated
#def fetch_player_data(player_id):
#    if type(player_id) != str:
#        player_id = str(player_id)
#    else:
#        pass
#
#    base_url = 'https://fantasy.premierleague.com/api/'
#    endpoint = f'element-summary/{player_id}/'
#    r = requests.get(base_url + endpoint).json()
#    stats = r['history']
#
#   return_dict = {}
#    for week in stats:
#        return_dict[week['round']] = {'points': week['total_points'],
#                                      'minutes': week['minutes']
#                                      }
#    return return_dict

element_id = '000'
endpoint = f'element-summary/{element_id}/'
r = requests.get(base_url + endpoint).json()

# extract info on positions

# extract info from teams - input = teams list pulled from FPL api
def parse_teams(teams):
    team_dict = {}
    for t in teams:
        team_dict[t['id']] = {'name': t['name'],
                              'strength': t['strength'],
                              'home_strength': t['strength_overall_home'],
                              'away_strength': t['strength_overall_away'],
                              'home_attack': t['strength_attack_home'],
                              'away_attack': t['strength_attack_away'],
                              'home_defense': t['strength_defence_home'],
                              'away_defense': t['strength_defence_away']}

    return team_dict

# add in info on opponent strength for each fixture
def add_opponent_stats(stats_df, team_dict):
    op_strength = []
    op_attack_strength = []
    op_defense_strength = []

    for ind, row in stats_df.iterrows():
        op = row['opponent_team']
        home = row['was_home']

        if home:
            strength = team_dict[op]['away_strength']
            attack = team_dict[op]['away_attack']
            defense = team_dict[op]['away_defense']
        else:
            strength = team_dict[op]['home_strength']
            attack = team_dict[op]['home_attack']
            defense = team_dict[op]['away_attack']

        op_strength.append(strength)
        op_attack_strength.append(attack)
        op_defense_strength.append(defense)

    stats_df['opponent_strength'] = op_strength
    stats_df['opponent_attack'] = op_attack_strength
    stats_df['opponent_defense'] = op_defense_strength

    return stats_df

# fetch stats on every single player performance
def fetch_player_data(player_id):
    if type(player_id) != str:
        player_id = str(player_id)
    else:
        pass

    base_url = 'https://fantasy.premierleague.com/api/'
    endpoint = f'element-summary/{player_id}/'
    r = requests.get(base_url + endpoint).json()
    stats = r['history']

    stats_list = []
    for week in stats:
        if week['minutes'] > 0:
            stats_list.append(week)

    return stats_list

# calculate form for each match (will do average of points from previous 4 matches - approximate FPL way of doing every 30 days)
def calculate_form(stats_df):
    players_with_actions = list(set(list(stats_df['element'])))
    rounds_for_form = 4 # keeping here as a settable variable
    form = []
    for p in players_with_actions:
        sub_df = stats_df[stats_df['element'] == p].reset_index().sort_values(by = ['round'])
        all_rounds = list(sub_df['round'])

        for ind, row in sub_df.iterrows():
            tp = 0
            round = row['round']
            first_round = round - rounds_for_form
            if first_round < 1:
                first_round = 1
            else:
                pass
            form_rounds = range(first_round, round)

            count = 0
            for r in form_rounds:
                try:
                    sub_sub_df = sub_df[sub_df['round'] == r].reset_index()
                    round_points = sub_sub_df['total_points'][0]
                    tp += round_points
                    count += 1
                except:
                    tp = tp
                    count = count

            try:
                f = tp/count
            except:
                f = 0
            form.append(f)

    stats_df['form'] = form

    return stats_df

# also calculate total points to date going into each fixture, some for influence,
# creativity, threat and ICT
def calculate_ict(stats_df):
    total_points = []
    points_avg = []
    total_inf = []
    total_cre = []
    total_trt = []
    total_ict = []
    players_with_actions = list(set(list(stats_df['element'])))
    for p in players_with_actions:
        sub_df = stats_df[stats_df['element'] == p].reset_index().sort_values(by=['round'])
        tp = 0
        inf = 0
        cre = 0
        trt = 0
        ict = 0
        count = 1
        for ind, row in sub_df.iterrows():
            total_points.append(tp)
            points_avg.append(tp / count)
            total_inf.append(inf / count)
            total_cre.append(cre / count)
            total_trt.append(trt / count)
            total_ict.append(ict / count)

            round_points = row['total_points']
            round_inf = float(row['influence'])
            round_cre = float(row['creativity'])
            round_trt = float(row['threat'])
            round_ict = float(row['ict_index'])

            tp += round_points
            inf += round_inf
            cre += round_cre
            trt += round_trt
            ict += round_ict

            count += 1

    for col, lst in zip(['preround_total_points', 'average_points', 'preround_influence', 'preround_creativity',
                         'preround_threat', 'preround_ict'],
                        [total_points, points_avg, total_inf, total_cre, total_trt, total_ict]):
        stats_df[col] = lst

    return stats_df

# reduce in-game transfers to a single number - using transfer_balance/selected - some selected are 0 so can't do a compact
def calculate_transfer_rate(stats_df):
    transfers = []
    for ind, row in stats_df.iterrows():
        selected = row['selected']
        if selected > 0:
            transfers.append(row['transfers_balance']/row['selected'])
        else:
            transfers.append(0)

    stats_df['transfers_reduced'] = transfers

    return stats_df

# implement fetch_player_data - input = players list pulled from FPL api
def fetch_gameweek_data(players, teams):
    position_dict = {1 : 'goalkeeper',
                     2: 'defender',
                     3: 'midfielder',
                     4: 'forward'}
    team_dict = parse_teams(teams)
    player_dict = parse_players(players)
    all_data = []
    for p in players:
        new_points = fetch_player_data(p['id'])
        if len(new_points) > 0:
            for np in new_points:
                all_data.append(np)

    i = 0
    out_dict = {}
    for d in all_data:
        out_dict[i] = d
        i += 1

    out_df = pd.DataFrame(data=out_dict).transpose()
    out_df['position'] = [position_dict[player_dict[i]['element_type']] for i in out_df['element']]

    # should I combine all these into a single functions
    out_df = add_opponent_stats(out_df, team_dict)
    out_df = calculate_form(out_df)
    out_df = calculate_ict(out_df)
    out_df = calculate_transfer_rate(out_df)

    return out_df, team_dict, player_dict

# extract info from players - Work in progress, dunno if need yet
def parse_players(players):
    player_dict = {}
    for p in players:
        id = p['id']
        player_dict[id] = {k:v for k, v in p.items() if k != 'id'}

    return player_dict




# future add(s) - 1. expected goals, 2. expected assists, 3. expected against, 4. goals scored?, 5. assists?




