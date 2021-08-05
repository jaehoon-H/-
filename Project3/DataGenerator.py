import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale

DATA_PATH = '../data/'

column_name = {
    'race_result': ['date', 'race_num', 'track_length', 'track_state', 'weather', 'rank', 'lane', 'horse', 'home',
                    'gender', 'age', 'weight', 'rating', 'jockey', 'trainer', 'owner', 'single_odds', 'double_odds'],
    'horse': ['horse', 'home', 'gender', 'birth', 'age', 'class', 'group', 'trainer', 'owner', 'father', 'mother',
              'race_count', 'first', 'second', '1yr_count', '1yr_first', '1yr_second', 'horse_money', 'rating',
              'price'],
}

# TODO: select columns to use
used_column_name = {
    'race_result': ['date', 'race_num', 'track_length', 'track_state', 'rank', 'horse', 'gender', 'age', 'weight','rating','double_odds'],
    'horse': ['date', 'horse', 'race_count', 'first', 'second', '1yr_count', '1yr_first', '1yr_second'],
}


def load_data():
    df_dict = dict()  # key: data type(e.g. jockey, trainer, ...), value: corresponding dataframe

    for data_type in ['horse', 'race_result']:
        fnames = sorted(os.listdir(DATA_PATH + data_type))
        df = pd.DataFrame()

        # concatenate all text files in the directory
        for fname in fnames:
            tmp = pd.read_csv(os.path.join(DATA_PATH, data_type, fname), header=None, sep=",",
                              encoding='cp949', names=column_name[data_type])

            if data_type != 'race_result':
                date = fname.split('.')[0]
                tmp['date'] = date[:4] + "-" + date[4:6] + "-" + date[-2:]

            df = pd.concat([df, tmp])

        # cast date column to dtype datetime
        df['date'] = df['date'].astype('datetime64[ns]')

        # append date offset to synchronize date with date of race_result data
        if data_type != 'race_result':
            df1 = df.copy()
            df1['date'] += pd.DateOffset(days=2)  # saturday
            df2 = df.copy()
            df2['date'] += pd.DateOffset(days=3)  # sunday
            df = df1.append(df2)

        # select columns to use
        df = df[used_column_name[data_type]]

        # insert dataframe to dictionary
        df_dict[data_type] = df

    ####### DO NOT CHANGE #######

    df_dict['race_result']['rank'].replace('1', 1., inplace=True)
    df_dict['race_result']['rank'].replace('2', 2., inplace=True)
    df_dict['race_result']['rank'].replace('3', 3., inplace=True)
    df_dict['race_result']['rank'].replace('4', 4., inplace=True)
    df_dict['race_result']['rank'].replace('5', 5., inplace=True)
    df_dict['race_result']['rank'].replace('6', 6., inplace=True)
    df_dict['race_result']['rank'].replace('7', 7., inplace=True)
    df_dict['race_result']['rank'].replace('8', 8., inplace=True)
    df_dict['race_result']['rank'].replace('9', 9., inplace=True)
    df_dict['race_result']['rank'].replace('10', 10., inplace=True)
    df_dict['race_result']['rank'].replace('11', 11., inplace=True)
    df_dict['race_result']['rank'].replace('12', 12., inplace=True)
    df_dict['race_result']['rank'].replace('13', 13., inplace=True)
    df_dict['race_result']['rank'].replace(' ', np.nan, inplace=True)

    # drop rows with rank missing values
    df_dict['race_result'].dropna(subset=['rank'], inplace=True)
    df_dict['race_result']['rank'] = df_dict['race_result']['rank'].astype('int')

    # make a column 'win' that indicates whether a horse ranked within the 3rd place
    df_dict['race_result']['win'] = df_dict['race_result'].apply(lambda x: 1 if x['rank'] < 4 else 0, axis=1)

    #################################

    # TODO: Make Features

    # 트랙 길이, 상태, 말의 부담 중량 0~1 scaling 및 전체 게임의 승률 계산, 2021-05-16 데이터까지만
    entire_mean = df_dict['race_result'][df_dict['race_result']['date'] <= '2021-05-16']['win'].mean()
    df_dict['race_result']['track_length'] = minmax_scale(df_dict['race_result']['track_length'])
    df_dict['race_result']['track_state'] = minmax_scale(df_dict['race_result']['track_state'])
    df_dict['race_result']['weight'] = minmax_scale(df_dict['race_result']['weight'])

    # 그 시점 직전까지 해당 말에 대한 배당률 평균 계산
    df_dict['race_result']['double_odds'] = 1/(1.25*df_dict['race_result']['double_odds'].astype('float'))
    df_dict['race_result']['double_odds'] = minmax_scale(df_dict['race_result']['double_odds'].astype('str'))

    horse_counts = df_dict['race_result']['horse'].drop_duplicates().tolist()
    date_count = df_dict['race_result']['date'].drop_duplicates().tolist()

    df_odds = pd.DataFrame()

    for horse in horse_counts:
        templist = []
        for racedate in date_count:
            if df_dict['race_result'][(df_dict['race_result']['date'] == racedate) & (df_dict['race_result']['horse'] == horse)]['double_odds'].size!=0:
                temp_odds=df_dict['race_result'][(df_dict['race_result']['date'] == racedate) & (df_dict['race_result']['horse'] == horse)][['date', 'horse','double_odds']]
                if len(templist) !=0:
                    temp_odds['double_odds_mean']=str(sum(templist)/len(templist))
                else:
                    temp_odds['double_odds_mean']=str(entire_mean)
                templist=templist+df_dict['race_result'][(df_dict['race_result']['date'] == racedate) & (df_dict['race_result']['horse'] == horse)]['double_odds'].astype('float').tolist()
                df_odds=df_odds.append(temp_odds)
    df_dict['race_result'] = df_dict['race_result'].merge(df_odds, on=['date', 'horse','double_odds'], how='left')

    # 성별에 따른 평균 승률 계산 및 계산 결과 0~1 scaling, 2021-05-16 데이터까지만
    df_female = df_dict['race_result'][(df_dict['race_result']['gender'] == '암') & (df_dict['race_result']['date'] <= '2021-05-16')]
    female_mean = df_female['win'].mean()
    df_dict['race_result']['gender'].replace('암', female_mean, inplace=True)
    df_male = df_dict['race_result'][(df_dict['race_result']['gender'] == '수') & (df_dict['race_result']['date'] <= '2021-05-16')]
    male_mean = df_male['win'].mean()
    df_dict['race_result']['gender'].replace('수', male_mean, inplace=True)
    df_other = df_dict['race_result'][(df_dict['race_result']['gender'] == '거') & (df_dict['race_result']['date'] <= '2021-05-16')]
    other_mean = df_other['win'].mean()
    df_dict['race_result']['gender'].replace('거', other_mean, inplace=True)
    df_dict['race_result']['gender'] = minmax_scale(df_dict['race_result']['gender'])

    # 나이에 따른 평균 승률 계산 및 계산 결과 0~1 scaling
    for i in range(15):
        a = str(i) + '세'
        tempmean = df_dict['race_result'][(df_dict['race_result']['age'] == a) & (df_dict['race_result']['date'] <= '2021-05-16')]['win'].mean()
        df_dict['race_result']['age'].replace(a, tempmean, inplace=True)
    df_dict['race_result']['age'] = minmax_scale(df_dict['race_result']['age'])

    #최근 1년간 1, 2위 비율 및 전체 기간 1, 2위 비율 계산
    df_dict['horse']['1yr_winning_prob'] = df_dict['horse'].\
        apply(lambda y: (y['1yr_first'] + y['1yr_second']) / (y['1yr_count']) if y['1yr_count'] >= 5 else (2 * entire_mean + y['1yr_first'] + y['1yr_second']) / (y['1yr_count'] + 2), axis=1)
    df_dict['horse']['winning_prob'] = df_dict['horse'].\
        apply(lambda y: (y['first'] + y['second']) / (y['race_count']) if y['race_count'] >= 5 else (2 * entire_mean + y['first'] + y['second']) / (y['race_count'] + 2), axis=1)

    #레이팅을 각 경기마다 다른 말과의 비교를 위해 0~1 scaling
    df_dict['race_result']['rating'].replace(' ', 0, inplace=True)
    date_count = df_dict['race_result']['date'].drop_duplicates().tolist()
    df_rated = pd.DataFrame()
    for race_date in date_count:
        ranum_count = df_dict['race_result'][df_dict['race_result']['date'] == race_date]['race_num'].drop_duplicates().tolist()
        for racenum in ranum_count:
            tempdf = df_dict['race_result'][(df_dict['race_result']['date'] == race_date) & (df_dict['race_result']['race_num'] == racenum)][['date', 'race_num', 'rating', 'horse']]
            if ((tempdf['rating'].astype('float') - tempdf['rating'].astype('float').mean()) ** 2).sum() == 0.0:
                tempdf['rated'] = str(1 - entire_mean)
            else:
                tempdf['rated'] = minmax_scale(tempdf['rating'])
            df_rated = df_rated.append(tempdf)
    df_dict['race_result'] = df_dict['race_result'].merge(df_rated, on=['date', 'race_num', 'rating', 'horse'],how='left')

    # drop duplicated rows

    # merge dataframes
    df = df_dict['race_result'].merge(df_dict['horse'], on=['date', 'horse'], how='left')

    # drop unnecessary columns which are used only for merging dataframes
    df.drop(['horse'], axis=1, inplace=True)

    df.to_csv('df_final.csv')
    return df


def get_data(test_day, is_training):
    if os.path.exists('df_final.csv'):
        print('preprocessed data exists')
        data_set = pd.read_csv('df_final.csv', index_col=0)
    else:
        print('preprocessed data NOT exists')
        print('loading data')
        data_set = load_data()

    # select training and test data by test day
    # TODO : cleaning or filling missing value
    training_data = data_set[~data_set['date'].isin(test_day)].fillna(0)
    test_data = data_set[data_set['date'].isin(test_day)].fillna(0)

    # TODO : make your input feature columns

    # select training x and y
    training_y = training_data['win']
    training_x = training_data.drop(['double_odds','win', 'date', 'race_num', 'rank', 'rating', 'race_count', 'first', 'second', '1yr_count', '1yr_first', '1yr_second'], axis=1)

    # select test x and y
    test_y = test_data['win']
    test_x = test_data.drop(['double_odds','win', 'date', 'race_num', 'rank', 'rating', 'race_count', 'first', 'second', '1yr_count', '1yr_first', '1yr_second'], axis=1)

    inspect_test_data(test_x, test_day)

    return (training_x, training_y) if is_training else (test_x, test_y)


def inspect_test_data(test_x, test_days):
    """
    Do not fix this function
    """
    df = pd.DataFrame()

    for test_day in test_days:
        fname = os.path.join(DATA_PATH, 'race_result', test_day.replace('-', '') + '.csv')
        tmp = pd.read_csv(fname, header=None, sep=",", encoding='cp949', names=column_name['race_result'])
        tmp.replace(' ', np.nan, inplace=True)
        tmp.dropna(subset=['rank'], inplace=True)

        df = pd.concat([df, tmp])

    # print(test_x.shape[0])
    # print(df.shape[0])

    assert test_x.shape[0] == df.shape[0], 'your test data is wrong!'


def main():
    get_data(['2019-04-20', '2019-04-21'], is_training=True)


if __name__ == '__main__':
    main()
