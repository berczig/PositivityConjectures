# the intention of this file is to understand the concept of value iteration
# using on a mdp where we know the transition and reward function
import numpy as np 
import random

class Value_Iteration:

    def __init__(self, n_states, n_actions, transitions, rewards, gamma):
        """
        n_states - number of states
        n_actions - numbers actions
        transitions - dictionary of positive probabilities (start state,action) |-> list of (target state, prop)
        rewards - dictionary of rewards (start state,action, end state) |-> reward
        """
        self.value = np.zeros(n_states) # init value function
        self.policy = {}
        self.n_states = n_states
        self.n_actions = n_actions
        self.transitions = transitions
        self.rewards = rewards
        self.gamma = gamma

    def optimize_value_function(self, maxsteps):
        for step in range(maxsteps+1):
            value_updated = np.zeros(self.n_states)
            for state in range(self.n_states):
                bestaction = 0
                bestvalue = 0
                for action in range(self.n_actions):
                    value = 0
                    for (target_state, prop) in self.transitions[(state, action)]:
                        # not being in self.rewards means the reward is 0 (why even bother adding a 0)
                        value += prop * (self.rewards.get((state, action, target_state), 0) + self.gamma*self.value[target_state])
                    if value > bestvalue:
                        bestvalue = value
                        bestaction = action 
                value_updated[state] = bestvalue
                self.policy[state] = bestaction
            #print("values:", step, self.value)
            if step < maxsteps: # the last step is only there to retrieve the action
                self.value = value_updated

class BridgeGame:
    LEFT = 0
    RIGHT = 1
    def __init__(self, n_bridges, n_streets, goal_reward, fall_chance, steps, gamma):
        self.n_bridges = n_bridges
        self.n_streets = n_streets
        self.goal_reward = goal_reward
        # Map looks like this GBBBBZSSSSSSSSSG, where Z is the start point, B a bridge, G  a goal and S a street
        n_states = n_bridges + n_streets + 3
        self.start_state = n_bridges+1
        n_actions = 2

        transitions = {}
        transitions[(0, self.RIGHT)] = [(1, 1)]
        transitions[(0, self.LEFT)] = [(0, 1)]
        transitions[(n_states-1, self.RIGHT)] = [(n_states-1, 1)]
        transitions[(n_states-1, self.LEFT)] = [(n_states-2, 1)]

        for i in range(1, n_states-1):
            if i <= n_bridges: # on a bridge you can fall down
                transitions[(i, self.RIGHT)] = [(i+1, 1-fall_chance), (self.start_state, fall_chance)]
                transitions[(i, self.LEFT)] = [(i-1, 1-fall_chance), (self.start_state, fall_chance)]
            else: # street or startpoint
                transitions[(i, self.RIGHT)] = [(i+1, 1)]
                transitions[(i, self.LEFT)] = [(i-1, 1)]
                
        rewards = {(0, self.LEFT, 0):goal_reward, (n_states-1, self.RIGHT, n_states-1):goal_reward}

        self.ValueIteration = Value_Iteration(n_states, n_actions, transitions, rewards, gamma)
        self.transitions = transitions

        self.transitions_sep = {}
        for key, value in transitions.items():
            self.transitions_sep[key] = list(zip(*value)) # [(state1, state2, ...), (prob1, prob2, ..)]
        self.rewards = rewards
        self.n_states = n_states
        self.ValueIteration.optimize_value_function(steps)


    def playgames(self, n_games, turns, policy, verbose = False):
        results = []
        for game in range(n_games):
            state = self.start_state
            reward = 0
            for turn in range(turns):
                action = policy[state] # policy gives an action
                if verbose:
                    print(game, turn, state, action)
                outcomes, probs = self.transitions_sep[(state, action)]
                newstate = random.choices(outcomes, probs, k=1)[0]
                reward += self.rewards.get((state, action, newstate), 0)
                state = newstate
            results.append(reward)
        return results


for streets in range(1, 20):
    B = BridgeGame(3, streets, 100, 0.5, 50, 0.9)
    print(streets, B.ValueIteration.policy[4])
    pol_left = dict([(state, B.LEFT) for state in range(B.n_states)])
    pol_right = dict([(state, B.RIGHT) for state in range(B.n_states)])
    meanscoreleft = np.mean(B.playgames(200, 2000, pol_left))
    meanscoreright = np.mean(B.playgames(200, 2000, pol_right))
    print(streets, meanscoreleft, meanscoreright)

#print(B.ValueIteration.value)
#print(B.ValueIteration.policy)
#print("results:", np.mean(B.playgames(20, 200, B.ValueIteration.policy)))