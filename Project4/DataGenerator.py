import os
import pandas as pd
import numpy as np

symbol_dict = {'cell': 'Celltrion',
               'hmotor': 'HyundaiMotor',
               'naver': 'NAVER',
               'kakao': 'Kakao',
               'lgchem': 'LGChemical',
               'lghnh': 'LGH_H',
               'bio': 'SamsungBiologics',
               'samsung1': 'SamsungElectronics',
               'samsung2': 'SamsungElectronics2',
               'sdi': 'SamsungSDI',
               'sk': 'SKhynix',
               'kospi': 'KOSPI',}

def symbol_to_path(symbol, base_dir="../data"):
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def merge_data(symbols):
    dates = pd.date_range('2010-01-01', '2021-05-17')

    df = pd.DataFrame(index=dates)
    for symbol in symbols:
        #네이버 액면 분할
        if symbol == 'NAVER':
            df_temp = pd.read_csv(symbol_to_path(symbol), index_col="Date", parse_dates=True,
                                  usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], na_values=['nan'])
            before=df_temp.loc[:'2018-10-11']
            before['Open']=before['Open']/5
            before['Volume']=before['Volume']*5
            after =df_temp.loc['2018-10-12':]
            df_temp=before.append(after)
            df_temp = df_temp.rename(columns={'Open': symbol + '_open', 'High': symbol + '_high', 'Low': symbol + '_low',
                         'Close': symbol + '_close', 'Volume': symbol + '_volume'})
        #카카오 액면 분할
        elif symbol == 'Kakao':
            df_temp = pd.read_csv(symbol_to_path(symbol), index_col="Date", parse_dates=True,
                                  usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], na_values=['nan'])
            # df_temp.loc[:'2018-10-11']['Open'].replace(df_temp.loc[:'2018-10-11']['Open']/5, inplace=True)
            before = df_temp.loc[:'2021-04-14']
            before['Open'] = before['Open'] / 5
            before['Volume'] = before['Volume'] * 5
            after = df_temp.loc['2021-04-15':]
            df_temp = before.append(after)
            df_temp = df_temp.rename(
                columns={'Open': symbol + '_open', 'High': symbol + '_high', 'Low': symbol + '_low',
                         'Close': symbol + '_close', 'Volume': symbol + '_volume'})

        else:
            df_temp = pd.read_csv(symbol_to_path(symbol), index_col="Date", parse_dates=True,
                                  usecols=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], na_values=['nan'])
            df_temp = df_temp.rename(columns={'Open': symbol + '_open', 'High': symbol + '_high', 'Low': symbol + '_low',
                                              'Close': symbol + '_close', 'Volume': symbol + '_volume'})
        df = df.join(df_temp)

    df = df.dropna(how='all')
    df = df.fillna(value=0)
    df = df.loc['2010-01-01':'2021-05-17']

    return df


def make_features(trade_company_list, start_date, end_date, is_training):

    # TODO: Choose symbols to make feature
    # symbol_list = ['Celltrion', 'HyundaiMotor', 'NAVER', 'Kakao', 'LGChemical', 'LGH&H',
    #                 'SamsungElectronics', 'SamsungElectronics2', 'SamsungSDI', 'SKhynix', 'KOSPI']
    #  feature_company_list = ['cell', 'hmotor', 'naver', 'kakao', 'lgchem', 'lghnh', 'bio', 'samsung1',
    #                         'sdi', 'sk', 'kospi']
    symbol_list = [symbol_dict[c] for c in trade_company_list]
    table = merge_data( symbol_list)

    # DO NOT CHANGE
    test_days = 10
    open_prices = np.asarray(table[[symbol_dict[c]+'_open' for c in trade_company_list]].loc[start_date:end_date])
    close_prices = np.asarray(table[[symbol_dict[c]+'_close' for c in trade_company_list]].loc[start_date:end_date])

    #print(open_prices[:,0])
    #print('----------------------')


    data = dict()
    for c in trade_company_list:
        data[c, 'close'] = table[symbol_dict[c] + '_close'].loc[start_date:end_date]
        data[c, 'open'] = table[symbol_dict[c] + '_open'].loc[start_date:end_date]
        data[c, 'diff_v'] = table[symbol_dict[c]+'_volume'].rolling(window=20).mean()-table[symbol_dict[c]+'_volume'].loc[start_date:end_date]
        data[c, 'ma5'] = table[symbol_dict[c] + '_close'].rolling(window=5).mean().loc[start_date:end_date]
        data[c, 'ma20'] = table[symbol_dict[c] + '_close'].rolling(window=20).mean().loc[start_date:end_date]
        data[c, 'ma60'] = table[symbol_dict[c] + '_close'].rolling(window=60).mean().loc[start_date:end_date]
        #data[c, 'ma_low_diff'] = (table[symbol_dict[c] + '_close'].rolling(window=60).mean()-table[symbol_dict[c]+'_low']).loc[start_date:end_date]
        data[c, 'bolband'] = table[symbol_dict[c] + '_close'].rolling(window=20).std().loc[start_date:end_date]*4
        data[c, 'macd_osc'] = (table[symbol_dict[c] + '_close'].rolling(window=12).mean()-\
                          table[symbol_dict[c] + '_close'].rolling(window=26).mean()-\
                          (table[symbol_dict[c] + '_close'].rolling(window=12).mean()-\
                          table[symbol_dict[c] + '_close'].rolling(window=26).mean()).rolling(window=9).mean()).loc[start_date:end_date]
        data[c, 'william_R'] = ((table[symbol_dict[c] + '_close'].rolling(window=15).max()-\
                               table[symbol_dict[c] + '_close'])/\
                               (table[symbol_dict[c] + '_close'].rolling(window=15).max()-
                                table[symbol_dict[c] + '_close'].rolling(window=15).min())).loc[start_date:end_date]

    input_days = 3

    features = list()
    final_tmps = []
    for a in range(data['bio', 'close'].shape[0] - input_days):
        tmps = list()
        # stock close price
        for c in trade_company_list:
            tmp = list(data[c, 'close'][a:a + input_days])
            tmps.append(tmp)
        # stock open price
        for i, c in enumerate(trade_company_list):
            tmp = list(data[c, 'open'][a:a + input_days])
            tmps[i].extend(tmp)
        # stock diff_v
        for i, c in enumerate(trade_company_list):
            tmp = list(data[c, 'diff_v'][a:a + input_days])
            tmps[i].extend(tmp)
        # stock ma
        for i, c in enumerate(trade_company_list):
            tmp = list(data[c, 'ma5'][a:a + input_days])
            tmps[i].extend(tmp)
        for i, c in enumerate(trade_company_list):
            tmp = list(data[c, 'ma20'][a:a + input_days])
            tmps[i].extend(tmp)
        for i, c in enumerate(trade_company_list):
            tmp = list(data[c, 'ma60'][a:a + input_days])
            tmps[i].extend(tmp)
        # stock bolband
        for i, c in enumerate(trade_company_list):
            tmp = list(data[c, 'bolband'][a:a + input_days])
            tmps[i].extend(tmp)
        # stock macd_osc
        for i, c in enumerate(trade_company_list):
            tmp = list(data[c, 'macd_osc'][a:a + input_days])
            tmps[i].extend(tmp)
        # stock william_R
        for i, c in enumerate(trade_company_list):
            tmp = list(data[c, 'william_R'][a:a + input_days])
            tmps[i].extend(tmp)
        final_tmps.append(tmps)
    features = np.array(final_tmps)


    if not is_training:
        return open_prices[-test_days:], close_prices[-test_days:], features[-test_days:]

    return open_prices[input_days:], close_prices[input_days:], features


if __name__ == "__main__":
    trade_company_list = ['hmotor', 'naver', 'kakao', 'lgchem', 'lghnh', 'bio', 'samsung1', 'sdi', 'sk']
    open, close, feature = make_features(trade_company_list, '2010-01-04', '2021-05-17', False)
    print(open,'\n')
    print(close,'\n')
    print(*feature[0],sep=' / ')
