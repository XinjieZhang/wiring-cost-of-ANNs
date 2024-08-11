# coding utf-8
from utils import task


# https://github.com/gyyang/multitask/blob/master/train.py

def get_default_hp(ruleset):
    """Get a defualt hp.

    :return:
        hp: a dictionary containing training hp parameters
    """
    num_ring = task.get_num_ring(ruleset)
    n_rule = task.get_num_rule(ruleset)

    n_eachring = 32
    n_input, n_output = 1 + num_ring * n_eachring + n_rule, n_eachring + 1
    hp = {
            # batch size for training
            'batch_size_train': 64,
            # batch_size for testing
            'batch_size_test': 1024,
            # input type: normal
            'in_type': 'normal',
            # Type of loss functions
            'loss_type': 'lsq',
            # Optimizer
            'optimizer': 'adam',
            # Type of activation functions, relu, softplus, tanh, elu
            'activation': 'softplus',
            # Time constant (ms)
            'tau': 100,
            # discretization time step (ms)
            'dt': 20,
            # discretization time step/time constant
            'alpha': 0.2,
            # recurrent noise
            'sigma_rec': 0.05,
            # input noise
            'sigma_x': 0.01,
            # leaky_rec weight initialization, diag, randortho, randgauss
            'w_rec_init': 'randortho',
            # l1 regularization on weight
            'l1_weight': 0,
            # l2 regularization on weight
            'l2_weight': 0,
            # Stopping performance
            'target_perf': 1.,
            # control the sparsity level of w_in, w_rec, w_out, None or float between (0,1)
            'epsilon_in': None,
            'epsilon_rec': None,
            'epsilon_out': None,
            # number of units each ring
            'n_eachring': n_eachring,
            # number of rings
            'num_ring': num_ring,
            # number of rules
            'n_rule': n_rule,
            # first input index for rule units
            'rule_start': 1+num_ring*n_eachring,
            # number of input units
            'n_input': n_input,
            # number of output units
            'n_output': n_output,
            # number of recurrent units
            'n_rnn': 256,
            # ruleset
            'ruleset': ruleset,
            # name to save
            'save_name': 'test',
            # learning rate
            'learning_rate': 0.001
            }

    return hp


def update_initializer(hp):
    if hp['activation'] == 'softplus':
        hp['w_in_start'] = 1.0
        hp['w_rec_start'] = 0.5
    elif hp['activation'] == 'tanh':
        hp['w_in_start'] = 1.0
        hp['w_rec_start'] = 1.0
    elif hp['activation'] == 'relu':
        hp['w_in_start'] = 1.0
        hp['w_rec_start'] = 0.5
    elif hp['activation'] == 'power':
        hp['w_in_start'] = 1.0
        hp['w_rec_start'] = 0.01
    elif hp['activation'] == 'retanh':
        hp['w_in_start'] = 1.0
        hp['w_rec_start'] = 0.5

