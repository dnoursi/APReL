import aprel
import numpy as np
import gym

from pprint import pprint

import random
from tqdm import tqdm
import copy
import pdb
import matplotlib.pyplot as plt


# https://medium.com/analytics-vidhya/multi-armed-bandit-analysis-of-epsilon-greedy-algorithm-8057d7087423
class EpsilonGreedyMAB:
    def __init__(self, epsilons, n_arms, explore_length):  # , counts, values):
        # first epsilon is for exploration, second is for after
        self.epsilons = epsilons
        self.initialize(n_arms)
        self.explore_length = explore_length
        self.update_count = 0
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
        epsilon = np.random.random()
        
        # If prob is not in epsilon, do exploitation of best arm so far
        if (epsilon > self.epsilons[self.update_count > self.explore_length]) and (
            sum(self.values) > 0
        ):
            # return np.argmax(self.values)
            return np.random.choice(
                a=len(self.values), p=self.values / np.sum(self.values)
            )
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
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
        
        self.update_count += 1
        
        return




def feature_func(traj):
    """Returns the features of the given MountainCar trajectory, i.e. \Phi(traj).

    Args:
        traj: List of state-action tuples, e.g. [(state0, action0), (state1, action1), ...]

    Returns:
        features: a numpy vector corresponding the features of the trajectory
    """
    states = np.array([pair[0] for pair in traj])
    actions = np.array([pair[1] for pair in traj[:-1]])
    min_pos, max_pos = states[:, 0].min(), states[:, 0].max()
    mean_speed = np.abs(states[:, 1]).mean()
    mean_vec = [-0.703, -0.344, 0.007]
    std_vec = [0.075, 0.074, 0.003]
    return (np.array([min_pos, max_pos, mean_speed]) - mean_vec) / std_vec


def main(u1_error):
    random.seed(7)

    env_name = "MountainCarContinuous-v0"
    gym_env = gym.make(env_name)

    np.random.seed(0)
    gym_env.seed(0)

    env = aprel.Environment(gym_env, feature_func)
    # env.render()
    trajectory_set = aprel.generate_trajectories_randomly(
        env, num_trajectories=15, max_episode_length=200, file_name=env_name, seed=0, headless=True
    )
    features_dim = len(trajectory_set[0].features)
    query_optimizer = aprel.QueryOptimizerDiscreteTrajectorySet(trajectory_set)

    params = {"weights": [0] * len(aprel.util_funs.get_random_normalized_vector(features_dim))}

    ### ###
    num_users = 2
    n_queries = 36
    bandit = EpsilonGreedyMAB(
        epsilons=[0.9, 0.5], n_arms=num_users, explore_length=int(n_queries / 3)
    )
    true_user = aprel.HumanUser(delay=0.5)

    user_model = aprel.SoftmaxUser(params)
    belief = aprel.SamplingBasedBelief(user_model, [], params)

    print("Estimated users parameters: ", str(belief.mean))

    ###


    user_choices = []
    rewards = []
    weightss = []
    bandit_weightss = []


    query = aprel.PreferenceQuery(trajectory_set[:2])
    for query_no in tqdm(range(n_queries)):
        user_choice = bandit.select_arm()
        user_choices.append(user_choice)

        print("\n\n\nnow query no", query_no, "user choice", user_choice)

        queries, objective_values = query_optimizer.optimize(
            "mutual_information", belief, query
        )

        print("Objective Value: " + str(objective_values[0]))

        k = 1
        responses = [
            int(
                queries[0].slate.trajectories[1].features[k]
                > queries[0].slate.trajectories[0].features[k]
            )
        ]
    
        if user_choice == 1 and np.random.random() < u1_error:
            responses[0] = 1-responses[0]
          # either 0 or 1
        # responses = true_user.respond(queries[0])

        #before = belief.mean["weights"]
        #print("Estimated user parameters before update: " + str(belief.mean))
    
        belief.update(aprel.Preference(queries[0], responses[0]))
        print("Estimated user parameters after update: " + str(belief.mean))
        weightss.append(copy.copy(belief.mean["weights"]))
        

        if len(weightss) > 1:
            reward = abs(np.dot(weightss[-2], weightss[-1]) / (np.linalg.norm(weightss[-2])* np.linalg.norm(weightss[-1])) )
            bandit.update(user_choice, reward)
            rewards.append(reward)
            
        bandit_weightss.append(copy.copy(bandit.values))

        dbgs = [
            (False, "bandit", bandit),
            (False, "belief", belief),
            (False, "user_model", user_model),
            (False, "true_user", true_user),
            (
                True,
                "belief",
                [
                    (k, len(vars(belief)[k]))
                    for k in vars(belief).keys()
                    if hasattr(vars(belief)[k], "__len__")
                ],
            ),
        ]

        print("entropy", (sum([np.exp(p) * p for p in belief.logprobs])))

        for dbg in dbgs:
            if dbg[0]:
                print(dbg[1], "params")
                if hasattr(dbg[2], "__dict__"):
                    pprint(vars(dbg[2]))
                else:
                    print(dbg[2])

        if False:
            print("bandit params are")
            pprint(vars(bandit))
            print("belief params are")
            pprint(vars(belief))
            print("usermodel params are")
            pprint(vars(user_model))
            print("user params are")
            pprint(vars(true_user))


    plt.plot(weightss)
    plt.legend(['min_pos','max_pos','velocity'])
    plt.title("belief params for error_rate="+str(u1_error))
    plt.savefig("belief.e="+str(u1_error)+"img.png")
    plt.clf()
    #plt.show()
    
    plt.plot(rewards)
    #plt.legend(['min_pos','max_pos','velocity'])
    plt.title("rewards for error_rate="+str(u1_error))
    plt.savefig("rewards.e="+str(u1_error)+"img.png")
    plt.clf()
    #plt.show()
    
    plt.plot(user_choices)
    plt.title("user_choices for error_rate="+str(u1_error))
    plt.savefig("user.e="+str(u1_error)+"img.png")
    plt.clf()
    #plt.show()

    plt.plot(bandit_weightss)
    plt.legend(['arm 0','arm 1'])
    plt.title("bandit_weights for error_rate="+str(u1_error))
    plt.savefig("banditweight.e="+str(u1_error)+"img.png")
    plt.clf()
    #plt.show()
    
    pdb.set_trace()
    
u1_errors = [0.01, 0.2, 0.3, 0.4, 0.6, 0.8]
for u1e in u1_errors:
    main(u1e)