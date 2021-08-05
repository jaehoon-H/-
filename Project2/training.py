from hmmlearn.hmm import GaussianHMM
import pickle
import DataGenerator


def main():

    start_date = '2020-01-01'
    end_date = '2021-04-01'

    training_x = DataGenerator.make_features(start_date, end_date, is_training=True)
    print(training_x.shape)
    # TODO: set model parameters
    n_components = 3
    model = GaussianHMM(n_components, covariance_type='full', n_iter=20)
    model.fit(training_x)

    # TODO: fix pickle file name
    filename = 'team03_model.pkl'
    pickle.dump(model, open(filename, 'wb'))
    print('saved {}'.format(filename))


if __name__ == "__main__":
    main()



