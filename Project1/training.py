from sklearn import neural_network as NN
import pickle
import DataGenerator


def main():
    start_date = '2010-01-01'
    end_date = '2020-04-06'

    training_x, training_y = DataGenerator.make_features(start_date, end_date, is_training=True)

    # TODO: set model parameters
    model = NN.MLPRegressor(max_iter=10000, hidden_layer_sizes=(32, 16,), activation='relu', solver='lbfgs',
                            learning_rate='adaptive', learning_rate_init=0.01, shuffle=False)
    model.fit(training_x, training_y)

    # TODO: fix pickle file name
    filename = 'team03_model.pkl'
    pickle.dump(model, open(filename, 'wb'))
    print('saved {}'.format(filename))


if __name__ == "__main__":
    main()





