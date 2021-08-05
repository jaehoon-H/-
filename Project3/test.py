import pickle
import DataGenerator
from sklearn.metrics import accuracy_score, recall_score, f1_score


def main():

    test_day = ['2021-05-08', '2021-05-09', '2021-05-15', '2021-05-16']
    # test_day = ['2020-02-09', '2020-02-15', '2020-02-16', '2020-02-22']
    test_x, test_y = DataGenerator.get_data(test_day, is_training=False)


    # TODO: fix pickle file name
    filename = 'team01_model.pkl'
    model = pickle.load(open(filename, 'rb'))
    print('load complete')
    print(model.get_params())

    # ================================ predict result ========================================
    pred_y = model.predict(test_x)
    print(pred_y)
    print('accuracy: {}'.format(accuracy_score(test_y, pred_y)))
    print('recall: {}'.format(recall_score(test_y, pred_y)))
    print('f1-score: {}'.format(f1_score(test_y, pred_y)))



if __name__ == '__main__':
    main()