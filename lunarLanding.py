import gym
import tensorflow as tf
import numpy as np
from collections import deque
from time import time, strftime, localtime
from datetime import datetime
import os
tf.get_logger().setLevel('ERROR')

# Define variables for Keras components
ModelBase = tf.keras.Model
layers = tf.keras.layers
regularizerBase = tf.keras.regularizers

# Global vars
STATE_SPACE_SIZE = 8
ACTION_SIZE = 4
MAX_EPISODES = 1000
MAX_EPISODE_LENGTH = 600
LOG_DIR = "./tf_train"
LOAD_DIR = "./good_run_save"
LOAD_EPISODE = 300


class Dqn(ModelBase):
    def __init__(self):
        super(Dqn, self).__init__()
        self.LIN_N = 128
        self.L1_N = 256
        self.LOUT_N = ACTION_SIZE
        # Input shape is the size of state [8]
        self.lIn = layers.Dense(self.LIN_N, activation=tf.nn.relu, name="Lin",
                                activity_regularizer=regularizerBase.l1_l2(l1=0.1, l2=0.01))
        self.l1 = layers.Dense(self.L1_N, activation=tf.nn.relu, name="L1",
                               activity_regularizer=regularizerBase.l1_l2(l1=0.1, l2=0.01))
        self.lOut = layers.Dense(self.LOUT_N, name="Lout")

    def shapeStr(self):
        return "%d->%d->%d" % (self.LIN_N, self.L1_N, self.LOUT_N)

    def call(self, x):
        x = self.lIn(x)
        x = self.l1(x)
        output = self.lOut(x)
        return output


class LanderBrain:

    def __init__(self, loadPath=None, loadEpisode=None):
        if loadPath is None and loadEpisode is None:
            self.GAMMA = 0.99
            self.EPS_MAX = 1.0
            self.EPS_MIN = 0.01
            self.ADJUSTER = 0.9995
            self.EXPERIENCE_RELAY_SIZE = 100000
            self.BATCH_SIZE = 512
            self.TARGET_NET_UPDATE_FREQ = 1
            self.SAVE_FREQ = 25
            self.WEIGHTS_FILE_NAME = "weights"
            now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            self.DIR_NAME = "{}/run-{}-log".format(LOG_DIR, now)
            self.LOG_FILE = None

            self.epsilon = self.EPS_MAX
            self.experienceRelay = deque(maxlen=self.EXPERIENCE_RELAY_SIZE)
            self.model = Dqn()
            self.target = Dqn()
            self.loss = tf.keras.losses.MeanSquaredError()
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            self.trainLoss = tf.keras.metrics.Mean(name='train_loss')
            self.tfLogger = self._initLogger()
        else:
            # Functions for testing
            self.WEIGHTS_FILE_NAME = "weights"
            self.DIR_NAME = loadPath

            self.model = Dqn()
            self._load(loadEpisode)

    def _reshapeState(self, state):
        return tf.reshape(state, [1, STATE_SPACE_SIZE])

    def testPolicy(self, state):
        output = self.model(self._reshapeState(state))
        return np.argmax(output)

    def _load(self, episode):
        try:
            epsStr = str(episode)
            folder = os.path.join(self.DIR_NAME, epsStr)
            fPath = os.path.join(folder, self.WEIGHTS_FILE_NAME)
            print("Starting to load model %s" % fPath)
            self.model.load_weights(fPath)
            print("Loaded")
        except:
            print("Model loading failure")
            quit()

    # Functions for training

    def _initLogger(self):
        if not os.path.isdir(LOG_DIR):
            os.mkdir(LOG_DIR)
        file_writer = tf.summary.create_file_writer(self.DIR_NAME)
        return file_writer

    def logLine(self, episode_number, stepTotal, lossTotal, score, epsilon):
        with self.tfLogger.as_default():
            # Add user custom data to TensorBoard
            lossPerStep = 0
            if stepTotal != 0:
                lossPerStep = lossTotal / float(stepTotal)
            tf.summary.scalar("Loss Per Step", lossPerStep,
                              step=episode_number)
            tf.summary.scalar("Total Loss", lossTotal, step=episode_number)
            tf.summary.scalar("Score", score, step=episode_number)
            tf.summary.scalar("Total Steps", stepTotal, step=episode_number)
            self.tfLogger.flush()

    def close(self, episode):
        self.LOG_FILE.close()
        self.save(episode)

    def policy(self, state):
        if np.random.sample() > self.epsilon:
            output = self.model(state[None, ])
            return np.argmax(output)
        return np.random.choice(ACTION_SIZE)

    def save(self, episode):
        if episode == 0 or episode % self.SAVE_FREQ != 0:
            return
        epsStr = str(episode)
        folder = os.path.join(self.DIR_NAME, epsStr)
        if os.path.isdir(folder):
            print("ALREADY SAVED, SKIPPING SAVE")
        os.mkdir(folder)
        self.model.save_weights(os.path.join(folder, self.WEIGHTS_FILE_NAME))

    def addToExpRelay(self, state, action, reward, nextState, isisFinished):
        self.experienceRelay.append(
            (state, action, reward, nextState, isisFinished))

    def getBatchFromExpRelay(self):
        if len(self.experienceRelay) < self.BATCH_SIZE:
            return None
        indices = np.arange(len(self.experienceRelay))
        chosenIndices = np.random.choice(indices, self.BATCH_SIZE)
        batch = np.array(list(self.experienceRelay))[chosenIndices]
        bState = batch[:, 0].tolist()
        bAction = batch[:, 1].tolist()
        bReward = batch[:, 2].tolist()
        bNextS = batch[:, 3].tolist()
        bIsisFinished = batch[:, 4].tolist()
        return (bState, bAction, bReward, bNextS, bIsisFinished)

    def updateTargetNet(self, episode):
        # No training done yet
        if not self.target.built:
            return
        if episode == 0 or episode % self.TARGET_NET_UPDATE_FREQ != 0:
            return
        self.target.set_weights(self.model.get_weights())

    def updateEpsilon(self):
        self.epsilon = max(self.EPS_MIN, self.epsilon * self.ADJUSTER)

    def getLabels(self, bReward, bNextS, bIsisFinished):
        # Get target Q
        targetPredictions = self.target(np.array(bNextS))
        labels = np.zeros(self.BATCH_SIZE)
        # Calculate target value
        # For the sampled actions, add the reward
        for i in range(self.BATCH_SIZE):
            labels[i] = bReward[i] + \
                (self.GAMMA *
                 np.max(targetPredictions[i]) if not bIsisFinished[i] else 0)
        return labels

    def train(self, bState, bAction, labels):
        with tf.GradientTape() as tape:
            # Get Q
            predictions = self.model(np.array(bState))
            predictions = tf.gather_nd(predictions, tf.stack(
                (tf.range(self.BATCH_SIZE), bAction), axis=1))
            # Take loss
            loss = self.loss(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        self.trainLoss(loss)


def train():
    env = gym.make('LunarLander-v2')
    moonLander = LanderBrain()

    for episode in range(MAX_EPISODES):
        episodeReward = 0
        state = env.reset()
        lastStepNum = 0
        for step in range(MAX_EPISODE_LENGTH):
            action = moonLander.policy(state)
            (nextState, stepReward, isFinished, _) = env.step(action)
            episodeReward += stepReward

            moonLander.addToExpRelay(
                state, action, stepReward, nextState, isFinished)
            batch = moonLander.getBatchFromExpRelay()
            if batch:
                (batchState, batchAction, batchReward,
                 batchNextState, batchIsFinished) = batch
                moonLander.train(batchState, batchAction, moonLander.getLabels(
                    batchReward, batchNextState, batchIsFinished))
                moonLander.updateEpsilon()
            if isFinished:
                lastStepNum = step
                break
            state = nextState
            lastStepNum = step
        moonLander.updateTargetNet(episode)
        moonLander.logLine(episode, lastStepNum,
                           float(str("%.3f" % moonLander.trainLoss.result())), episodeReward, moonLander.epsilon)
        # moonLander.logLine("%d, %.5f, %.5f, %.5f, %d" % (
        #     episode, moonLander.epsilon, episodeReward, moonLander.trainLoss.result(), lastStepNum))
        print("Episode %4s, Epsilon %5s, Reward %6s, Loss %5s, T %d" % (episode, str("%.4f" % moonLander.epsilon), str(
            "%.3f" % episodeReward), str("%.3f" % moonLander.trainLoss.result()), lastStepNum))
        moonLander.save(episode)
    moonLander.close(episode)
    env.close()


def test():
    env = gym.make('LunarLander-v2')
    moonLander = LanderBrain(LOAD_DIR, LOAD_EPISODE)

    for episode in range(MAX_EPISODES):
        episodeReward = 0
        state = env.reset()
        lastStepNum = 0
        for step in range(MAX_EPISODE_LENGTH):
            env.render()
            action = moonLander.testPolicy(state)
            (nextState, stepReward, isFinished, _) = env.step(action)
            episodeReward += stepReward

            if isFinished:
                lastStepNum = step
                break
            state = nextState

        print("Episode: %5s, Reward: %5s, Total Steps:  %d" %
              (episode, str("%.3f" % episodeReward), lastStepNum))
    env.close()


if __name__ == "__main__":
    # train()
    test()
