import pickle
from car_racing import *
import numpy as np
from multiprocessing import Process, Queue
import random

n_threads      = 3
population    = 100
mutation_rate = 1
n_ctrl_pts    = 8
mlp_layers    = (n_ctrl_pts + 6, 12, 10, 4)
sigma         = 0.14
laps          = 1
task_queue, result_queue = Queue(), Queue()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

class multilayerperceptron:
    def __init__(self, layer_sizes, activation='relu'):
        assert len(layer_sizes) >= 2
        assert activation in ['sigmoid', 'relu']
        self.activation = sigmoid if activation == 'sigmoid' else relu
        self.ws = []
        for a, b in zip(layer_sizes[:-1], layer_sizes[1:]):
            mat = np.zeros((a + 1, b + 1))
            mat[-1, -1] = np.infty if activation == 'sigmoid' else 1
            mat[:, :-1] = np.random.rand(*mat[:, :b].shape) * 0.5 - 0.25
            self.ws.append(mat)

    def mutate(self, mutation_rate, sigma):
        for i, mat in enumerate(self.ws):
            p = np.clip(mutation_rate / np.prod(mat[:, :-1].shape), 0, 1)
            self.ws[i][:, :-1] += np.random.choice([0, 1], size=mat[:, :-1].shape, p=[1-p, p]) *\
                                  np.random.normal(*mat[:, :-1].shape) * sigma
    def crossover(self, target):
        start_index = random.randint(0, len(self.ws)-1)
        end_index = random.randint(start_index, len(self.ws))
        for i, mat in enumerate(self.ws):
            if start_index < i < end_index:
                jj = self.ws[i][:, :-1]
                self.ws[i][:, :-1] = target.ws[i][:, :-1]
                target.ws[i][:, :-1] = jj
        return target

    def feedforward(self, x):
        x = np.append(x, 1)
        for w in self.ws:
            x = self.activation(x @ w)
            assert x[-1] == 1
        x[0] -= x[-2]
        return x[:-2]


def fitness(env, dna):
    _, _, done, state  = env.fast_reset()
    
    step = 1
    max_reward = 0
    max_reward_step = 0
    while state.on_road and state.laps < laps:
        if step - max_reward_step > 4.5*FPS:
            break
        reaction = dna.feedforward(state.as_array(n_ctrl_pts))
        _, _, done, state = env.step(reaction)

        if state.reward > max_reward:
            max_reward = state.reward
            max_reward_step = step
        step += 1
    return max_reward


def fitness_run(env, dna):
    _, _, _, state = env.fast_reset()
    step = 1
    max_reward = 0
    max_reward_step = 0
    while state.on_road and state.laps < 2:
        if step - max_reward_step > 4.5 * FPS:
            break
        reaction = dna.feedforward(state.as_array(n_ctrl_pts))
        _, _, _, state = env.step(reaction)
        if state.reward > max_reward:
            max_reward = state.reward
            max_reward_step = step
        step += 1
        env.render()
    return max_reward

def measure_fitness(dnas):
    fitnesses = np.array([-100.0] * len(dnas))
    for item in enumerate(dnas):
        task_queue.put(item)
    for _ in range(len(dnas)):
        i, fit = result_queue.get()
        fitnesses[i] = fit
    return fitnesses

def process_entry(task_queue, result_queue):
    penv = CarRacing()
    penv.reset()

    while True:
        i, dna = task_queue.get()
        result_queue.put((i, fitness(penv, dna)))


dnas = None
epoch = 0
fitnesses = np.array([0] * population)
indices = list(range(population))
history = []

def initialize_dnas():
    global dnas, epoch, history
    try:
        with open('saved_dnas', 'rb') as f:
            saved = pickle.load(f)
            if saved['mlp_layers'] != mlp_layers:
                raise Exception('Incompatible MLP layers')
            dnas, epoch, history = saved['dnas'], saved['epoch'], saved['history']
            pop_diff = saved['pop'] - population
            if   pop_diff > 0:
                dnas = dnas[:population]
            elif pop_diff < 0:
                dnas.extend([dnas[-1]] * pop_diff)
            print('Loaded saved dnas')
    except IOError as e:
        dnas = np.array([multilayerperceptron(mlp_layers) for _ in range(population)])
        print('Generating new dnas')
    except BaseException as e:
        dnas = np.array([multilayerperceptron(mlp_layers) for _ in range(population)])
        print('Generating new dnas due to error:', str(e))


def mp_n_tournaments(n):
    global dnas, fitnesses, indices
    replacements = 0
    np.random.shuffle(indices)
    children = [copy.deepcopy(dnas[i]) for i in indices[:n]]
    for child in children:
        child.mutate(mutation_rate, sigma)
    for index, child in enumerate(children):
        random_index = 0
        children[random_index] = child.crossover(children[random_index])
        children[index] = child
    children_fit = measure_fitness(children)
    for child, child_fit in zip(children, children_fit):
        other = np.random.randint(population)
        if child_fit > fitnesses[other]:
            dnas[other], fitnesses[other] = child, child_fit
            replacements += 1
    return replacements

def replace_with_fittest(fittest):
    global dnas, fitnesses
    for i in range(population):
        if i != fittest:
            dnas[i] = copy.deepcopy(dnas[fittest])
    fitnesses = measure_fitness(dnas)


if __name__ == '__main__':
    processes = [Process(target=process_entry, args=(task_queue, result_queue)) for _ in range(n_threads)]
    for p in processes:
        p.start()
    initialize_dnas()
    fitnesses = measure_fitness(dnas)
    if epoch == 0:
        random_dna = np.random.randint(population)
        last_fittest_dna = copy.deepcopy(dnas[random_dna])
    else:
        last_fittest_dna = copy.deepcopy(dnas[-1])
    print('     |           Fitnesses                  |     Replacements')
    print('{:4} |  {:>8} {:>8} {:>8} {:>8} |'.format('Iter', '100 %ile', '75 %ile', '50 %ile', '25 %ile'))
    f = open('saved_dnas', 'wb', buffering=0)
    while True:
        replacements = mp_n_tournaments(population)
        idx = np.argsort(fitnesses)
        fitnesses = fitnesses[idx]
        dnas = dnas[idx]
        history.append(fitnesses)
        epoch += 1
        if epoch % 10 == 0:
            print('{:4}    {:8.2f} {:8.2f} {:8.2f} {:8.2f}       {:4.1f}%'.format(
                epoch, fitnesses[-1], fitnesses[population * 75 // 100], fitnesses[population * 50 // 100],
                fitnesses[population * 25 // 100], 100 * replacements / population)
            )
            f.seek(0)
            pickle.dump(
                    {'dnas': dnas,
                    'epoch': epoch,
                    'mlp_layers': mlp_layers,
                    'pop': population,
                    'last_fittest_dna': last_fittest_dna,
                    'history': history},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL)
            f.truncate()
            env = CarRacing()
            env.reset()
            fit = fitness_run(env, dnas[-1])
            env.close()
            last_fittest_dna = copy.deepcopy(dnas[-1])
        if epoch == 100:
            exit(0)



