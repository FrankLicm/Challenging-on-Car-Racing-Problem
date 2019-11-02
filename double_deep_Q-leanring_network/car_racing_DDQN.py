import random, math, gym
import numpy as np
import keras
import keras.optimizers as Kopt
from keras.layers import Convolution2D
from keras.layers import Dense, Flatten, Input
from keras.models import Model
from keras import backend as K
import pickle
import os

models_path = ""
data_path = ""
RBG_mode = True
Conv_folder_2_gray = False
temporal_buffer = 4
load_and_test = False
load_and_train = False
train_env_render = False
enable_epsilon_rst = True
green_sc_plty = 0.4
action_repeat = 8
max_nb_episodes = int(10e3)
max_nb_step = action_repeat * 100
learning_rate = 0.001
huber_loss_delta = 1.0
memory_capacity = int(1e4)
batch_size = 120
max_reward = 10
dropout_thr = 0.1
env_seed = 2
image_width = 96
image_height = 96
image_stack = 4
image_size = (image_width, image_height, image_stack)
gamma = 0.99
max_epsilon = 1
min_epsilon = 0.02
exploration_stop = int(max_nb_step * 10)
_lambda = - math.log(0.001) / exploration_stop
update_target_frequency = int(200)
action_buffer = np.array([
                    [0.0, 0.0, 0.0],
                    [-0.3, 0.03, 0.0],
                    [0.4, 0.03, 0.0],
                    [0.0, 0.2, 0.0]])
NumberOfDiscActions = len(action_buffer)

class Agent_Model:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        self.model_ = self._createModel()  # target network

        self.ModelsPath_cp = models_path + "DDQN_model_cp.h5"
        self.ModelsPath_cp_per = models_path + "DDQN_model_cp_per.h5"

        save_best = keras.callbacks.ModelCheckpoint(self.ModelsPath_cp,
                                                monitor='loss',
                                                verbose=1,
                                                save_best_only=True,
                                                mode='min',
                                                period=20)
        save_per = keras.callbacks.ModelCheckpoint(self.ModelsPath_cp_per,
                                                monitor='loss',
                                                verbose=1,
                                                save_best_only=False,
                                                mode='min',
                                                period=400)

        self.callbacks_list = [save_best, save_per]


    def _createModel(self):
        agent_model_in = Input(shape=self.stateCnt, name='agent_model_in')
        x = agent_model_in
        x = Convolution2D(16, (16,16), strides=(2,2), activation='relu')(x)
        x = Convolution2D(32, (8,8), strides=(2,2), activation='relu')(x)
        x = Flatten(name='flattened')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(self.actionCnt, activation="linear")(x)
        model = Model(inputs=agent_model_in, outputs=x)
        self.opt = Kopt.RMSprop(lr=learning_rate)
        model.compile(loss=huber_loss, optimizer=self.opt)
        return model


    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=(batch_size // 2), epochs=epochs, verbose=verbose, callbacks=self.callbacks_list)


    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)


    def predictOne(self, s, target=False):
        x = s[np.newaxis,:,:,:]
        return self.predict(x, target)


    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())

class Memory:
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

class Agent:
    steps = 0
    epsilon = max_epsilon
    memory = Memory(memory_capacity)

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.agent_model = Agent_Model(stateCnt, self.actionCnt)
        self.no_state = np.zeros(stateCnt)
        self.x = np.zeros((batch_size,) + image_size)
        self.y = np.zeros([batch_size, self.actionCnt])
        self.errors = np.zeros(batch_size)
        self.rand = False

        self.agentType = 'Learning'
        self.maxEpsilone = max_epsilon

    def act(self, s):
        if random.random() < self.epsilon:
            best_act = np.random.randint(self.actionCnt)
            self.rand=True
            return Select_Action(best_act), Select_Action(best_act)
        else:
            act_soft = self.agent_model.predictOne(s)
            best_act = np.argmax(act_soft)
            self.rand=False
            return Select_Action(best_act), act_soft

    def observe(self, sample):
        x, y, errors = self._getTargets([(0, sample)])
        self.memory.add(errors[0], sample)

        if self.steps % update_target_frequency == 0:
            self.agent_model.updateTargetModel()
        self.steps += 1
        self.epsilon = min_epsilon + (self.maxEpsilone - min_epsilon) * math.exp(-_lambda * self.steps)

    def _getTargets(self, batch):
        states = np.array([ o[1][0] for o in batch ])
        states_ = np.array([ (self.no_state if o[1][3] is None else o[1][3]) for o in batch ])
        p = agent.agent_model.predict(states)
        p_ = agent.agent_model.predict(states_, target=False)
        pTarget_ = agent.agent_model.predict(states_, target=True)
        act_ctr = np.zeros(self.actionCnt)

        for i in range(len(batch)):
            o = batch[i][1]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]

            a = Select_ArgAction(a)
            t = p[i]
            act_ctr[a] += 1
            oldVal = t[a]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + gamma * pTarget_[i][np.argmax(p_[i])]  # double DQN

            self.x[i] = s
            self.y[i] = t
            if self.steps % 20 == 0 and i == len(batch)-1:
                print('t',t[a], 'r: %.4f' % r,'mean t',np.mean(t))
                print ('act ctr: ', act_ctr)
            self.errors[i] = abs(oldVal - t[a])
        return (self.x, self.y, self.errors)

    def replay(self):
        batch = self.memory.sample(batch_size)
        x, y, errors = self._getTargets(batch)
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])

        self.agent_model.train(x, y)

class RandomAgent:
    memory = Memory(memory_capacity)
    exp = 0
    steps = 0

    def __init__(self, actionCnt):
        self.actionCnt = actionCnt
        self.agentType = 'Random'
        self.rand = True

    def act(self, s):
        best_act = np.random.randint(self.actionCnt)
        return Select_Action(best_act), Select_Action(best_act)

    def observe(self, sample):  # in (s, a, r, s_) format
        error = abs(sample[2])  # reward
        self.memory.add(error, sample)
        self.exp += 1
        self.steps += 1

    def replay(self):
        pass

class Environment:
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)
        self.env.seed(5)
        from gym import envs
        envs.box2d.car_racing.WINDOW_H = 500
        envs.box2d.car_racing.WINDOW_W = 600

        self.episode = 0
        self.reward = []
        self.step = 0
        self.action_stuck = 0

    def run(self, agent):
        self.env.seed(env_seed)
        img = self.env.reset()
        img =  rgb2gray(img, True)
        s = np.zeros(image_size)
        for i in range(image_stack):
            s[:,:,i] = img

        s_ = s
        R = 0
        self.step = 0

        a_soft = a_old = np.zeros(agent.actionCnt)
        a = action_buffer[0]
        while True:
            if agent.agentType=='Learning' :
                if train_env_render == True :
                    self.env.render('human')


            if self.step % action_repeat == 0:

                if agent.rand == False:
                    a_old = a_soft

                a, a_soft = agent.act(s)
                if enable_epsilon_rst:
                    if agent.rand == False:
                        if a_soft.argmax() == a_old.argmax():
                            self.action_stuck += 1
                            if self.action_stuck >= 200:
                                agent.steps = 0
                                agent.agent_model.opt.lr.set_value(learning_rate * 10)

                                self.action_stuck = 0
                        else:
                            self.action_stuck = max(self.action_stuck -2, 0)

                img_rgb, r, done, info = self.env.step(a)

                if not done:
                    img = rgb2gray(img_rgb, True)
                    for i in range(image_stack - 1):
                        s_[:,:,i] = s_[:,:,i+1]
                    s_[:,:, image_stack - 1] = img

                else:
                   s_ = None

                R += r
                r = (r / max_reward)
                if np.mean(img_rgb[:,:,1]) > 185.0:
                    r -= green_sc_plty

                r = np.clip(r, -1 ,1)

                agent.observe( (s, a, r, s_) )
                agent.replay()
                s = s_

            else:
                img_rgb, r, done, info = self.env.step(a)
                if not done:
                    img =  rgb2gray(img_rgb, True)
                    for i in range(image_stack - 1):
                        s_[:,:,i] = s_[:,:,i+1]
                    s_[:,:, image_stack - 1] = img
                else:
                   s_ = None
                R += r
                s = s_

            if (self.step % (action_repeat * 5) == 0) and (agent.agentType == 'Learning'):
                print('step:', self.step, 'R: %.1f' % R, a, 'rand:', agent.rand)

            self.step += 1

            if done or (R<-5) or (self.step > max_nb_step) or np.mean(img_rgb[:, :, 1]) > 185.1:
                self.episode += 1
                self.reward.append(R)
                print('Done:', done, 'R<-5:', (R<-5), 'Green>185.1:',np.mean(img_rgb[:,:,1]))
                break

        print("\n Episode ", self.episode,"/", max_nb_episodes, agent.agentType)
        print("Avg Episode R:", R/self.step, "Total R:", sum(self.reward))

    def test(self, agent):
        self.env.seed(env_seed)
        img= self.env.reset()
        img = rgb2gray(img, True)
        s = np.zeros(image_size)
        for i in range(image_stack):
            s[:,:,i] = img

        R = 0
        self.step = 0
        done = False
        while True :
            self.env.render('human')
            if self.step % action_repeat == 0:
                if(agent.agentType == 'Learning'):
                    act1 = agent.agent_model.predictOne(s)
                    act = Select_Action(np.argmax(act1))
                else:
                    act = agent.act(s)

                if self.step <= 8:
                    act = Select_Action(3)

                img_rgb, r, done, info = self.env.step(act)
                img = rgb2gray(img_rgb, True)
                R += r

                for i in range(image_stack - 1):
                    s[:,:,i] = s[:,:,i+1]
                s[:,:, image_stack - 1] = img

            if(self.step % 10) == 0:
                print('Step:', self.step, 'action:',act, 'R: %.1f' % R)
                print(np.mean(img_rgb[:,:,0]), np.mean(img_rgb[:,:,1]), np.mean(img_rgb[:,:,2]))
            self.step += 1

            if done or (R<-5) or (agent.steps > max_nb_step) or np.mean(img_rgb[:, :, 1]) > 185.1:
                R = 0
                self.step = 0
                print('Done:', done, 'R<-5:', (R<-5), 'Green>185.1:',np.mean(img_rgb[:,:,1]))
                break

class SumTree:
    write = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

def rgb2gray(rgb, norm):
    gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    if norm:
        # normalize
        gray = gray.astype('float32') / 128 - 1

    return gray


def save_DDQL(Path, Name, agent, R):
    if not os.path.exists(Path):
        os.makedirs(Path)
    agent.brain.model.save(Path + Name)
    dump_pickle(agent.memory, Path + Name + 'Memory')
    dump_pickle([agent.epsilon, agent.steps, agent.brain.opt.get_config()], Path + Name + 'AgentParam')
    dump_pickle(R, Path + Name + 'Rewards')


def dump_pickle(obj, name):
    with open(name, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(name):
    with open(name, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def Select_Action(Act):
    return action_buffer[Act]


def Select_ArgAction(Act):
    for i in range(NumberOfDiscActions):
        if np.all(Act == action_buffer[i]):
            return i
    raise ValueError('Select_ArgAction: Act not in action_buffer')


def huber_loss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) <= huber_loss_delta
    if cond == True:
        loss = 0.5 * K.square(err)

    else:
        loss = 0.5 * huber_loss_delta ** 2 + huber_loss_delta * (K.abs(err) - huber_loss_delta)
    return K.mean(loss)


if __name__ == "__main__":
    env = Environment('CarRacing-v0')
    stateCnt = image_size
    actionCnt = env.env.action_space.shape[0]
    random.seed(901)
    np.random.seed(1)
    agent = Agent(stateCnt, NumberOfDiscActions)
    randomAgent = RandomAgent(NumberOfDiscActions)
    if load_and_test == False:
        if load_and_train == False:
            while randomAgent.exp < memory_capacity:
                env.run(randomAgent)
                print(randomAgent.exp, "/", memory_capacity)

            agent.memory = randomAgent.memory
            randomAgent = None

            print("Start learning")

            while env.episode < max_nb_episodes:
                env.run(agent)

            save_DDQL(models_path, "DDQN_model.h5", agent, env.reward)
        else:
            print('Load pre-trained agent and learn')
            agent.agent_model.model.load_weights(models_path + "DDQN_model.h5")
            agent.agent_model.updateTargetModel()
            try:
                agent.memory = load_pickle(models_path + "DDQN_model.h5" + "Memory")
                Params = load_pickle(models_path + "DDQN_model.h5" + "AgentParam")
                agent.epsilon = Params[0]
                agent.steps = Params[1]
                opt = Params[2]
                agent.agent_model.opt.decay.set_value(opt['decay'])
                agent.agent_model.opt.epsilon = opt['epsilon']
                agent.agent_model.opt.lr.set_value(opt['lr'])
                agent.agent_model.opt.rho.set_value(opt['rho'])
                env.reward = load_pickle(models_path + "DDQN_model.h5" + "Rewards")
                del Params, opt
            except:
                while randomAgent.exp < memory_capacity:
                    env.run(randomAgent)
                    print(randomAgent.exp, "/", memory_capacity)
                agent.memory = randomAgent.memory
                randomAgent = None
                agent.maxEpsilone = max_epsilon / 5
            print("Start learning")
            while env.episode < max_nb_episodes:
                env.run(agent)

            save_DDQL(models_path, "DDQN_model.h5", agent, env.reward)
    else:
        print('Load agent and play')
        agent.agent_model.model.load_weights(models_path + "DDQN_model.h5")
        done_ctr = 0
        while done_ctr < 5:
            env.test(agent)
            done_ctr += 1
        env.env.close()
    agent.agent_model.model.save(models_path + "DDQN_model.h5")