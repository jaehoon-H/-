import os
import pandas as pd


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
    dates = pd.date_range(start_date, end_date, freq='D')
    df = pd.DataFrame(index=dates)

    if 'USD_KRW' not in symbols:
        symbols.insert(0, 'USD_KRW')

    for symbol in symbols:
        df_temp = pd.read_csv(get_data_path(symbol), index_col="Date", parse_dates=True, na_values=['nan'])
        df_temp = df_temp.reindex(dates)
        df_temp.columns = [symbol + '_' + col for col in df_temp.columns]  # rename columns
        df = df.join(df_temp)

    return df

def correlation_analysis(start_date, end_date):
    table = merge_data(start_date, end_date, symbols=['AUD_KRW', 'CNY_KRW', 'EUR_KRW', 'GBP_KRW', 'HKD_KRW', 'JPY_KRW'])
    table2 = merge_data(start_date, end_date, symbols=['Brent Oil', 'Copper', 'Crude Oil WTI', 'Gasoline', 'Gold', 'Natural Gas', 'Platinum', 'Silver'])
    table.dropna(axis=0, inplace=True)
    table2.dropna(axis=0, inplace=True)
    for col in table:
        for idx, d in enumerate(table[col]):
            try:
                table[col][idx] = float(d.replace(',', '').replace('%', ''))
            except AttributeError:
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


    # main_column = table['USD_KRW_Price']
    # target_column1 = []
    # target_column2 = []
    #
    # for col in table:
    #     target_column1.append(col)
    # for col in table2:
    #     target_column2.append(col)
    #
    # for col in target_column1:
    #     print('USD_KRW_Price/'+col)
    #     print(main_column.corr(table[col], method = 'pearson'))
    # for col in target_column2:
    #     print('USD_KRW_Price/'+col)
    #     print(main_column.corr(table2[col], method = 'pearson'))


def make_features(start_date, end_date, is_training):
    # TODO: select symbols
    # commodity : BrentOil, Copper, CrudeOil, Gasoline, Gold, NaturalGas, Platinum, Silver
    # currency : AUD_KRW, CNY_KRW, EUR_KRW, GBP_KRW, HKD_KRW, JPY_KRW, USD_KRW
    table = merge_data(start_date, end_date, symbols=['HKD_KRW'])
    table2 = merge_data(start_date, end_date, symbols=['Brent Oil'])
    # TODO: cleaning or filling missing value
    table.dropna(axis=0, inplace=True)
    table2.dropna(axis=0, inplace=True)

    for col in table:
        for idx, d in enumerate(table[col]):
            try:
                table[col][idx] = float(d.replace(',','').replace('%',''))
            except AttributeError:
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


    # TODO: select columns to use
    input_days = 10

    USD_price = table['USD_KRW_Price'].astype('float64')
    HKD_price = table['HKD_KRW_Price']
    Brent_price = table2['Brent Oil_Price']

    USD_price = USD_price[USD_price.index.isin(Brent_price.index)]
    HKD_price = HKD_price[HKD_price.index.isin(Brent_price.index)]
    Brent_price = Brent_price

    y = windowing_y(USD_price, input_days)

    USD_price = (USD_price - USD_price.mean()) / USD_price.std()
    HKD_price = (HKD_price - HKD_price.mean()) / HKD_price.std()
    Brent_price = (Brent_price - Brent_price.mean()) / Brent_price.std()

    USD_x = windowing_x(USD_price, input_days)
    HKD_x = windowing_x(HKD_price, input_days)
    Brent_x = windowing_x(Brent_price, input_days)

    x = []

    for i in range(len(USD_x)):
        x.append(USD_x[i].append(HKD_x[i]).append(Brent_x[i]))


    # split training and test data
    training_x = x[:-10]
    training_y = y[:-10]
    test_x = x[-10]
    test_y = y[-10]

    return (training_x, training_y) if is_training else (test_x, test_y)


def windowing_y(data, input_days):
    windows = [data[i + input_days:i + input_days + 10] for i in range(len(data) - input_days)]
    return windows


def windowing_x(data, input_days):
    windows = [data[i:i + input_days] for i in range(len(data) - input_days)]
    return windows


if __name__ == "__main__":
    start_date = '2010-01-01'
    end_date = '2021-04-01'

    make_features(start_date, end_date, is_training=False)
    #correlation_analysis(start_date, end_date)
