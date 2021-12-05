import aprel
import numpy as np
import gym

from pprint import pprint

import random
         
# https://medium.com/analytics-vidhya/multi-armed-bandit-analysis-of-epsilon-greedy-algorithm-8057d7087423
class EpsilonGreedyMAB():
    def __init__(self, epsilons, n_arms, explore_length): #, counts, values):
        # first epsilon is for exploration, second is for after
        self.epsilons = epsilons
        self.initialize(n_arms)
        self.explore_length = explore_length
#        self.counts = counts # Count represent counts of pulls for each arm. For multiple arms, this will be a list of counts.
#        self.values = values # Value represent average reward for specific arm. For multiple arms, this will be a list of values.
        return 
    
    # Initialise k number of arms
    def initialize(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        return
    
    # Epsilon greedy arm selection
    def select_arm(self):
        epsilon = random.random()
        print(epsilon)
        # If prob is not in epsilon, do exploitation of best arm so far
        if (epsilon > self.epsilons[len(self.values) > self.explore_length]) and (sum(self.values) > 0):
            # return np.argmax(self.values)
            return np.random.choice(a=len(self.values), p=self.values/np.sum(self.values))
        # If prob falls in epsilon range, do exploration
        else:
            return random.randrange(len(self.values))
    
    # Choose to update chosen arm and reward
    def update(self, chosen_arm, reward):
        # update counts pulled for chosen arm
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]
        
        # Update average/mean value/reward for chosen arm
        value = self.values[chosen_arm]
        new_value = ((n-1)/float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
        return


env_name = 'MountainCarContinuous-v0'
gym_env = gym.make(env_name)

np.random.seed(0)
gym_env.seed(0)

def feature_func(traj):
    """Returns the features of the given MountainCar trajectory, i.e. \Phi(traj).

    Args:
        traj: List of state-action tuples, e.g. [(state0, action0), (state1, action1), ...]

    Returns:
        features: a numpy vector corresponding the features of the trajectory
    """
    states = np.array([pair[0] for pair in traj])
    actions = np.array([pair[1] for pair in traj[:-1]])
    min_pos, max_pos = states[:,0].min(), states[:,0].max()
    mean_speed = np.abs(states[:,1]).mean()
    mean_vec = [-0.703, -0.344, 0.007]
    std_vec = [0.075, 0.074, 0.003]
    return (np.array([min_pos, max_pos, mean_speed]) - mean_vec) / std_vec


env = aprel.Environment(gym_env, feature_func)
# env.render()
trajectory_set = aprel.generate_trajectories_randomly(env, num_trajectories=5,
                                                      max_episode_length=120,
                                                      file_name=env_name, seed=0)
features_dim = len(trajectory_set[0].features)
query_optimizer = aprel.QueryOptimizerDiscreteTrajectorySet(trajectory_set)

params = {'weights': aprel.util_funs.get_random_normalized_vector(features_dim)}

### ###
num_users = 2
n_queries = 12
bandit = EpsilonGreedyMAB(epsilons = [0.9, 0.5], n_arms = num_users, explore_length=int(n_queries/3))
true_user = aprel.HumanUser(delay=0.5) for _ in range(num_users)]

user_models = [aprel.SoftmaxUser(params) for _ in range(num_users)]
beliefs = [aprel.SamplingBasedBelief(user_model, [], params) for user_model in user_models]

print('Estimated users parameters: ', *[str(belief.mean) for belief in beliefs])

###
query = aprel.PreferenceQuery(trajectory_set[:2])
for query_no in range(n_queries):
    user_choice = bandit.select_arm()
    
    print('now query no', query_no, 'user choice', user_choice)
    
    true_user=true_users[user_choice]
    belief=beliefs[user_choice]

    queries, objective_values = query_optimizer.optimize('mutual_information', belief, query)
        
    print('Objective Value: ' + str(objective_values[0]))
    responses = true_user.respond(queries[0])
    before = belief.mean["weights"]
    print('Estimated user parameters before update: ' + str(belief.mean))
    belief.update(aprel.Preference(queries[0], responses[0]))
    after = belief.mean["weights"]
    print('Estimated user parameters after update: ' + str(belief.mean))
    
    rwrd = np.linalg.norm(before - after)
    bandit.update(user_choice, rwrd)
    
    
    print('bandit params are')
    pprint(vars(bandit))
    
    
    
