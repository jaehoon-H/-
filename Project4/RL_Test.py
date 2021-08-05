import numpy as np
import DataGenerator as DataGenerator
from decision_ql import QLearningDecisionPolicy
from simulation import do_action


def run(policy, initial_budget, initial_num_stocks, open_prices, close_prices, features):

    budget = initial_budget
    num_stocks = initial_num_stocks

    for i in range(len(open_prices)):
        current_state = np.asmatrix(np.hstack((features[i], [budget])))
        action = policy.select_action(current_state, is_training=False)
        budget, num_stocks = do_action(action, budget, num_stocks, open_prices[i])
        print('Day {}'.format(i+1))
        print('action {} / budget {} / shares {}'.format(action, budget, [num_stocks]))
        print('portfolio with  open price : {}'.format(budget + num_stocks * open_prices[i]))
        print('portfolio with close price : {}\n'.format(budget + num_stocks * close_prices[i]))

    portfolio = budget + num_stocks * close_prices[-1]

    print('Finally, you have')
    print('budget: %.2f won' % budget)
    print('Share : {}'.format(num_stocks))
    print('Share value : {} won'.format(close_prices[-1]))
    print()

    return portfolio


if __name__ == '__main__':
    start, end = '2021-03-01', '2021-05-17'
    company_list = ['hmotor', 'naver', 'lgchem', 'lghnh', 'bio', 'sdi', 'sk']
    actions = ['buying', 'not_buying']
    #########################################
    open_prices, close_prices, features = DataGenerator.make_features(company_list, start, end, is_training=False)
    total_budget = 10. ** 8
    budget = 10. ** 7
    num_stocks = 0
    input_dim = len(features[0,0])+1
    final_portfolio = 0
    for i, c in enumerate(company_list):
        policy = QLearningDecisionPolicy(0, 1, 0, actions, input_dim, c)
        final_portfolio += run(policy, budget, num_stocks, open_prices[:,i], close_prices[:,i], features[:,i])
    final_portfolio += (10. ** 7) * (10-len(company_list))
    print("Final portfolio: %.2f won" % final_portfolio)

