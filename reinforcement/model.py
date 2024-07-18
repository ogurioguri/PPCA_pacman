
"""
Functions you should use.
Please avoid importing any other torch functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""

from torch.nn import Module
from torch.nn import  Linear
from torch import tensor, double, optim
from torch.nn.functional import relu, mse_loss



class DeepQNetwork(Module):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim
        super(DeepQNetwork, self).__init__()
        # Remember to set self.learning_rate, self.numTrainingGames,
        # and self.batch_size!
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.01
        self.numTrainingGames = 11000
        self.batch_size = 32
        self.parameters = []
        self.layer1 = Linear(state_dim, 500)
        self.layer2 = Linear(500, 500)
        self.layer3 = Linear(500, 500)
        self.layer4 = Linear(500, action_dim)
        self.parameters.extend(self.layer1.parameters())
        self.parameters.extend(self.layer2.parameters())
        self.parameters.extend(self.layer3.parameters())
        self.parameters.extend(self.layer4.parameters())



        "**END CODE"""
        self.double()


    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"
        Q_values = self.forward(states)
        return mse_loss(Q_values, Q_target)


    def forward(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            result: (batch_size x num_actions) numpy array of Q-value
                scores, for each of the actions
        """
        "*** YOUR CODE HERE ***"
        x = tensor(states, dtype=double)
        x = relu(self.layer1(x))
        x = relu(self.layer2(x))
        x = relu(self.layer3(x))
        x = self.layer4(x)
        return x
    

    
    def run(self, states):
        return self.forward(states)

    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        You can look at the ML project for an idea of how to do this, but note that rather
        than iterating through a dataset, you should only be applying a single gradient step
        to the given datapoints.

        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """ 
        "*** YOUR CODE HERE ***"
        optimizer = optim.SGD(self.parameters, lr=self.learning_rate)
        optimizer.zero_grad()
        loss = self.get_loss(states, Q_target)
        loss.backward()
        optimizer.step()   