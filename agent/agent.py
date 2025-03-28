import numpy as np
from abstract_classes import BaseAgent

class QLearningAgent(BaseAgent):
    def agent_init(self, agent_info={}):
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))
        self.epsilon = agent_info.get("epsilon")
        self.step_size = agent_info.get("step_size")
        self.discount = agent_info.get("discount")
        self.num_states = agent_info.get("num_states")
        self.num_actions = agent_info.get("num_actions")
        self.q = np.zeros((self.num_states, self.num_actions))

    def agent_start(self, state):
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(self.q[state, :])

        self.prev_state = state
        self.prev_action = action
        return self.prev_action

    def agent_step(self, reward, state):
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(self.q[state, :])

        td_target = reward + self.discount * np.max(self.q[state, :])
        q_s_a = self.q[self.prev_state, self.prev_action]
        self.q[self.prev_state, self.prev_action] = q_s_a + self.step_size * (td_target - q_s_a)

        self.prev_state = state
        self.prev_action = action
        return action

    def agent_end(self, reward):
        td_target = reward
        q_s_a = self.q[self.prev_state, self.prev_action]
        self.q[self.prev_state, self.prev_action] = q_s_a + self.step_size * (td_target - q_s_a)

    def agent_cleanup(self):
        self.prev_state = None

    def argmax(self, q_values):
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []
            if q_values[i] == top:
                ties.append(i)
        return self.rand_generator.choice(ties)
    def agent_message(self):
        pass


class DoubleQLearningAgent(BaseAgent):
    def agent_init(self, agent_info={}):
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))
        self.epsilon = agent_info.get("epsilon")
        self.step_size = agent_info.get("step_size")
        self.discount = agent_info.get("discount")
        self.num_states = agent_info.get("num_states")
        self.num_actions = agent_info.get("num_actions")

        self.q1 = np.zeros((self.num_states, self.num_actions))
        self.q2 = np.zeros((self.num_states, self.num_actions))

    def agent_start(self, state):
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(self.q1[state, :] + self.q2[state, :])

        self.prev_state = state
        self.prev_action = action
        return self.prev_action

    def agent_step(self, reward, state):
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(self.q1[state, :] + self.q2[state, :])

        if self.rand_generator.rand() < 0.5:
            td_target = reward + self.discount * self.q2[state, self.argmax(self.q1[state, :])]
            self.q1[self.prev_state, self.prev_action] += self.step_size * (td_target - self.q1[self.prev_state, self.prev_action])
        else:
            td_target = reward + self.discount * self.q1[state, self.argmax(self.q2[state, :])]
            self.q2[self.prev_state, self.prev_action] += self.step_size * (td_target - self.q2[self.prev_state, self.prev_action])

        self.prev_state = state
        self.prev_action = action
        return action

    def agent_end(self, reward):
        if self.rand_generator.rand() < 0.5:
            self.q1[self.prev_state, self.prev_action] += self.step_size * (reward - self.q1[self.prev_state, self.prev_action])
        else:
            self.q2[self.prev_state, self.prev_action] += self.step_size * (reward - self.q2[self.prev_state, self.prev_action])

    def agent_cleanup(self):
        self.prev_state = None

    def argmax(self, q_values):
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []
            if q_values[i] == top:
                ties.append(i)
        return self.rand_generator.choice(ties)
    def agent_message(self):
        pass


class TripleQLearningAgent(BaseAgent):
    def agent_init(self, agent_info={}):
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))
        self.epsilon = agent_info.get("epsilon")
        self.step_size = agent_info.get("step_size")
        self.discount = agent_info.get("discount")
        self.num_states = agent_info.get("num_states")
        self.num_actions = agent_info.get("num_actions")

        self.q1 = np.zeros((self.num_states, self.num_actions))
        self.q2 = np.zeros((self.num_states, self.num_actions))
        self.q3 = np.zeros((self.num_states, self.num_actions))

    def agent_start(self, state):
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(self.q1[state, :] + self.q2[state, :] + self.q3[state, :])

        self.prev_state = state
        self.prev_action = action
        return self.prev_action

    def agent_step(self, reward, state):
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(self.q1[state, :] + self.q2[state, :] + self.q3[state, :])

        table_choice = self.rand_generator.choice([1, 2, 3])
        if table_choice == 1:
            td_target = reward + self.discount * max(self.q2[state, self.argmax(self.q1[state, :])], self.q3[state, self.argmax(self.q1[state, :])])
            self.q1[self.prev_state, self.prev_action] += self.step_size * (td_target - self.q1[self.prev_state, self.prev_action])
        elif table_choice == 2:
            td_target = reward + self.discount * max(self.q1[state, self.argmax(self.q2[state, :])], self.q3[state, self.argmax(self.q2[state, :])])
            self.q2[self.prev_state, self.prev_action] += self.step_size * (td_target - self.q2[self.prev_state, self.prev_action])
        else:
            td_target = reward + self.discount * max(self.q1[state, self.argmax(self.q3[state, :])], self.q2[state, self.argmax(self.q3[state, :])])
            self.q3[self.prev_state, self.prev_action] += self.step_size * (td_target - self.q3[self.prev_state, self.prev_action])

        self.prev_state = state
        self.prev_action = action
        return action

    def agent_end(self, reward):
        table_choice = self.rand_generator.choice([1, 2, 3])
        if table_choice == 1:
            self.q1[self.prev_state, self.prev_action] += self.step_size * (reward - self.q1[self.prev_state, self.prev_action])
        elif table_choice == 2:
            self.q2[self.prev_state, self.prev_action] += self.step_size * (reward - self.q2[self.prev_state, self.prev_action])
        else:
            self.q3[self.prev_state, self.prev_action] += self.step_size * (reward - self.q3[self.prev_state, self.prev_action])

    def agent_cleanup(self):
        self.prev_state = None

    def argmax(self, q_values):
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []
            if q_values[i] == top:
                ties.append(i)
        return self.rand_generator.choice(ties)
    
    def agent_message(self):
        pass


class QuadrupleQLearningAgent(BaseAgent):
    def agent_init(self, agent_info={}):
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))
        self.epsilon = agent_info.get("epsilon")
        self.step_size = agent_info.get("step_size")
        self.discount = agent_info.get("discount")
        self.num_states = agent_info.get("num_states")
        self.num_actions = agent_info.get("num_actions")

        self.q1 = np.zeros((self.num_states, self.num_actions))
        self.q2 = np.zeros((self.num_states, self.num_actions))
        self.q3 = np.zeros((self.num_states, self.num_actions))
        self.q4 = np.zeros((self.num_states, self.num_actions))

    def agent_start(self, state):
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(self.q1[state, :] + self.q2[state, :] + self.q3[state, :] + self.q4[state, :])

        self.prev_state = state
        self.prev_action = action
        return self.prev_action

    def agent_step(self, reward, state):
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(self.q1[state, :] + self.q2[state, :] + self.q3[state, :] + self.q4[state, :])

        table_choice = self.rand_generator.choice([1, 2, 3, 4])
        if table_choice == 1:
            td_target = reward + self.discount * max(self.q2[state, self.argmax(self.q1[state, :])], 
                                                     self.q3[state, self.argmax(self.q1[state, :])], 
                                                     self.q4[state, self.argmax(self.q1[state, :])])
            self.q1[self.prev_state, self.prev_action] += self.step_size * (td_target - self.q1[self.prev_state, self.prev_action])
        elif table_choice == 2:
            td_target = reward + self.discount * max(self.q1[state, self.argmax(self.q2[state, :])], 
                                                     self.q3[state, self.argmax(self.q2[state, :])], 
                                                     self.q4[state, self.argmax(self.q2[state, :])])
            self.q2[self.prev_state, self.prev_action] += self.step_size * (td_target - self.q2[self.prev_state, self.prev_action])
        elif table_choice == 3:
            td_target = reward + self.discount * max(self.q1[state, self.argmax(self.q3[state, :])], 
                                                     self.q2[state, self.argmax(self.q3[state, :])], 
                                                     self.q4[state, self.argmax(self.q3[state, :])])
            self.q3[self.prev_state, self.prev_action] += self.step_size * (td_target - self.q3[self.prev_state, self.prev_action])
        else:
            td_target = reward + self.discount * max(self.q1[state, self.argmax(self.q4[state, :])], 
                                                     self.q2[state, self.argmax(self.q4[state, :])], 
                                                     self.q3[state, self.argmax(self.q4[state, :])])
            self.q4[self.prev_state, self.prev_action] += self.step_size * (td_target - self.q4[self.prev_state, self.prev_action])

        self.prev_state = state
        self.prev_action = action
        return action

    def agent_end(self, reward):
        table_choice = self.rand_generator.choice([1, 2, 3, 4])
        if table_choice == 1:
            self.q1[self.prev_state, self.prev_action] += self.step_size * (reward - self.q1[self.prev_state, self.prev_action])
        elif table_choice == 2:
            self.q2[self.prev_state, self.prev_action] += self.step_size * (reward - self.q2[self.prev_state, self.prev_action])
        elif table_choice == 3:
            self.q3[self.prev_state, self.prev_action] += self.step_size * (reward - self.q3[self.prev_state, self.prev_action])
        else:
            self.q4[self.prev_state, self.prev_action] += self.step_size * (reward - self.q4[self.prev_state, self.prev_action])

    def agent_cleanup(self):
        self.prev_state = None

    def argmax(self, q_values):
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []
            if q_values[i] == top:
                ties.append(i)
        return self.rand_generator.choice(ties)
    
    def agent_message(self):
        pass
