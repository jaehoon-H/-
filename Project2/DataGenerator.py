import os
import pandas as pd
import numpy as np


def get_data_path(symbol):
    # Return CSV file path given symbol.
    commodity_dir = '../data/commodities'
    currency_dir = '../data/currencies'

    if symbol in ['AUD_KRW', 'CNY_KRW', 'EUR_KRW', 'GBP_KRW', 'HKD_KRW', 'JPY_KRW', 'USD_KRW']:
        path = os.path.join(currency_dir, symbol + '.csv')
    else:
        path = os.path.join(commodity_dir, symbol + '.csv')

    return path


def merge_data(start_date, end_date, symbols):
    dates = pd.date_range(start_date, end_date)
    df = pd.DataFrame(index=dates)

    if 'Gold' not in symbols:
        symbols.insert(0, 'Gold')

    for symbol in symbols:
        df_temp = pd.read_csv(get_data_path(symbol), index_col="Date", parse_dates=True, na_values=['nan'])
        df_temp.columns = [symbol + '_' + col for col in df_temp.columns]  # rename columns
        df = df.join(df_temp)

    return df

def correlation_analysis(start_date, end_date):
    table = merge_data(start_date, end_date, symbols=['USD_KRW', 'AUD_KRW', 'CNY_KRW', 'EUR_KRW', 'GBP_KRW', 'HKD_KRW', 'JPY_KRW'])
    table2 = merge_data(start_date, end_date, symbols=['Brent Oil', 'Copper', 'Crude Oil WTI', 'Gasoline', 'Natural Gas', 'Platinum', 'Silver'])
    table.dropna(axis=0, inplace=True)
    table2.dropna(axis=0, inplace=True)
    for col in table:
        if col[-4:] == 'Vol.':
            tmp_df = table[table[col] != '-']  # 거래량 - 으로 표현된 부분은 중앙값으로 처리
            for idx, item in enumerate(tmp_df[col]):
                tmp_df[col][idx] = float(item.replace('K', '').replace('M', ''))
            median = tmp_df[col].median()
            for idx, item in enumerate(table[col]):
                table[col][idx] = item.replace('K', '').replace('M', '').replace('-', '{}'.format(median))
                table[col][idx] = float(table[col][idx])
        else:
            for idx, item in enumerate(table[col]):
                try:
                    table[col][idx] = item.replace(',', '').replace('K', '').replace('%', '')
                    table[col][idx] = float(table[col][idx])
                except AttributeError:
                    pass
                except ValueError:
                    pass
    for col in table2:
        if col[-4:] == 'Vol.':
            tmp_df = table2[table2[col] != '-']  # 거래량 - 으로 표현된 부분은 중앙값으로 처리
            for idx, item in enumerate(tmp_df[col]):
                tmp_df[col][idx] = float(item.replace('K', '').replace('M', ''))
            median = tmp_df[col].median()
            for idx, item in enumerate(table2[col]):
                table2[col][idx] = item.replace('K', '').replace('M', '').replace('-', '{}'.format(median))
                table2[col][idx] = float(table2[col][idx])
        else:
            for idx, item in enumerate(table2[col]):
                try:
                    table2[col][idx] = item.replace(',', '').replace('K', '').replace('%', '')
                    table2[col][idx] = float(table2[col][idx])
                except AttributeError:
                    pass
                except ValueError:
                    pass

    table = table.astype('float64')
    table2 = table2.astype('float64')

    print(table.corr(method = 'pearson'))
    print(table2.corr(method = 'pearson'))

def make_features(start_date, end_date, is_training):
    # TODO: select symbols
    # commodity : BrentOil, Copper, CrudeOil, Gasoline, Gold, NaturalGas, Platinum, Silver
    # currency : AUD, CNY, EUR, GBP, HKD, JPY, USD
    table = merge_data(start_date, end_date, symbols=['JPY_KRW'])
    table2 = merge_data(start_date, end_date, symbols=['Gold', 'Silver', 'Copper'])

    # TODO: cleaning or filling missing value
    table.dropna(inplace=True)
    table2.dropna(axis=0, inplace=True)

    for col in table:
        if col[-4:] == 'Vol.':
            tmp_df = table[table[col] != '-']  # 거래량 - 으로 표현된 부분은 중앙값으로 처리
            for idx, item in enumerate(tmp_df[col]):
                tmp_df[col][idx] = float(item.replace('K', '').replace('M', ''))
            median = tmp_df[col].median()
            for idx, item in enumerate(table[col]):
                table[col][idx] = item.replace('K', '').replace('M', '').replace('-', '{}'.format(median))
                table[col][idx] = float(table[col][idx])
        else:
            for idx, item in enumerate(table[col]):
                try:
                    table[col][idx] = item.replace(',', '').replace('K', '').replace('%', '')
                    table[col][idx] = float(table[col][idx])
                except AttributeError:
                    pass
                except ValueError:
                    pass
    for col in table2:
        if col[-4:] == 'Vol.':
            tmp_df = table2[table2[col] != '-']  # 거래량 - 으로 표현된 부분은 중앙값으로 처리
            for idx, item in enumerate(tmp_df[col]):
                tmp_df[col][idx] = float(item.replace('K', '').replace('M', ''))
            median = tmp_df[col].median()
            for idx, item in enumerate(table2[col]):
                table2[col][idx] = item.replace('K', '').replace('M', '').replace('-', '{}'.format(median))
                table2[col][idx] = float(table2[col][idx])
        else:
            for idx, item in enumerate(table2[col]):
                try:
                    table2[col][idx] = item.replace(',', '').replace('K', '').replace('%', '')
                    table2[col][idx] = float(table2[col][idx])
                except AttributeError:
                    pass
                except ValueError:
                    pass

    input_days = 10

    Gold_price = table2['Gold_Price'].astype('float64')
    JPY_price = table['JPY_KRW_Price'].astype('float64')
    Silver_price = table2['Silver_Price'].astype('float64')
    Copper_price = table2['Copper_Price'].astype('float64')

    JPY_price = JPY_price[JPY_price.index.isin(Gold_price.index)]
    Gold_price_ = Gold_price[Gold_price.index.isin(JPY_price.index)]
    Silver_price = Silver_price[Silver_price.index.isin(JPY_price.index)]
    Copper_price = Copper_price[Copper_price.index.isin(JPY_price.index)]


    # TODO:  make features
    Gold_diff = np.diff(Gold_price_)
    Gold_price_ = Gold_price_[1:]
    Jpy_diff = np.diff(JPY_price)
    Silver_diff = np.diff(Silver_price)
    Copper_diff = np.diff(Copper_price)
    training_sets = list()
    for time in range(len(Gold_price_)-input_days):
        gold_diff = Gold_diff[time:time + input_days]
        jpy_diff = Jpy_diff[time:time + input_days]
        silver_diff = Silver_diff[time:time + input_days]
        copper_diff = Copper_diff[time:time + input_days]
        daily_feature = np.concatenate((gold_diff[::-1], silver_diff[::-1], jpy_diff[::-1], copper_diff[::-1]))

        training_sets.append(daily_feature)

    training_sets = np.array(training_sets)

    training_x = training_sets[:-10]

    test_x = training_sets[-10:]

    past_price = Gold_price[-11:-1]
    target_price = Gold_price[-10:]

    return training_x if is_training else (test_x, past_price, target_price)


if __name__ == "__main__":
    start_date = '2020-01-01'
    end_date = '2021-04-01'

    make_features(start_date, end_date, is_training=False)
    #correlation_analysis(start_date, end_date)\