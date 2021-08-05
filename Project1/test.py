import DataGenerator
import pickle
import numpy as np
import pandas as pd
from DataGenerator import get_data_path
from sklearn.metrics import mean_absolute_error


def get_test_dollar_price(start_date, end_date):
    """
    Do not fix this function
    """
    df = pd.read_csv(get_data_path('USD_KRW'), index_col="Date", parse_dates=True, na_values=['nan'])
    price = pd.to_numeric(df['Price'].loc[end_date: start_date][:10][::-1].apply(lambda x: x.replace(',', '')))
    return price


def main():
    start_date = '2010-01-01'
    end_date = '2021-04-01'

    test_x, test_y = DataGenerator.make_features(start_date, end_date, is_training=False)
    print(get_test_dollar_price(start_date, end_date))
    ###################################################################################################################
    # inspect test data
    assert test_y.tolist() == get_test_dollar_price(start_date, end_date).tolist(), 'your test data is wrong!'
    ###################################################################################################################

    # TODO: fix pickle file name
    filename = 'team03_model.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))
    print('load complete')
    print(loaded_model.get_params())

    predict = loaded_model.predict([test_x])
    print(predict)
    print('mae: ', mean_absolute_error(np.reshape(predict, -1), test_y))


if __name__ == '__main__':
    main()
