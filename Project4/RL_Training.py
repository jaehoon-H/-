from decision_ql import QLearningDecisionPolicy
import DataGenerator as DataGenerator
import simulation as simulation
import tensorflow as tf
tf.compat.v1.reset_default_graph()

if __name__ == '__main__':
    start, end = '2020-03-14', '2021-05-17'

    company_list = ['hmotor', 'naver', 'lgchem', 'lghnh', 'bio', 'sdi', 'sk']
    #company_list  = ['sk', 'kakao', 'hmotor']
    #company_list = ['lgchem', 'samsung1']

    actions = ['buying', 'not_buying']

    epsilon = 0.99
    gamma = 0.4
    lr = 0.01
    num_epoch = 20
    #########################################
    open_prices, close_prices, features = DataGenerator.make_features(company_list, start, end, is_training=True)
    total_budget = 10. ** 8
    budget = 10.**7
    num_stocks = 0
    input_dim = len(features[0,0])+1
    policy_dict = {}
    for c in company_list:
        policy_dict[c] = QLearningDecisionPolicy(epsilon=epsilon, gamma=gamma, lr=lr, actions=actions, input_dim=input_dim,
                                     model_dir= "model")
    for i, c in enumerate(company_list):
        policy = policy_dict[c]
        simulation.run_simulations(company_list=company_list, policy=policy, budget=budget, num_stocks=num_stocks,
                               open_prices=open_prices[:,i], close_prices=close_prices[:,i], features=features[:,i],
                               num_epoch=num_epoch,c=c  )
        policy.save_model(c)


