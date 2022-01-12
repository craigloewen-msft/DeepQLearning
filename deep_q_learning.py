# Do imports

from re import I
import gym
import random
import numpy as np
import tensorflow as tf
from keras import layers
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Model

from collections import deque
from tensorflow.keras.optimizers import RMSprop
from keras import backend as K
from datetime import datetime
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras.models import clone_model
import os.path
import time
from timeit import default_timer as timer
from datetime import timedelta

# Flag class
class ParametersClass:
    def __init__(self):
        self.train_dir = 'tf_train_breakout'
        # self.restore_file_path = './tf_train_breakout/breakout_model_20180610205843_36h_12193ep_sec_version.h5' 
        self.restore_file_path = './tf_train_breakout/trained_model.h5'
        self.num_episode = 800
        self.observe_step_num = 1000
        self.epsilon_step_num = 10000
        self.refresh_target_model_num = 10000
        self.replay_memory = 300000
        self.regularizer_scale = 0.01
        self.batch_size = 512
        self.learning_rate = 0.00025
        self. init_epsilon = 1.0
        self.final_epsilon = 0.1
        self.gamma = 0.99
        self.resume = False
        self.render = True

FLAGS = ParametersClass()

ATARI_SHAPE = 8
ACTION_SIZE = 4

class LandingBrain:

    def __init__(self):
        self.model = self.create_dqn_model()
        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.trainLoss = tf.keras.metrics.Mean(name='train_loss')
        self.target = self.create_dqn_model()

    def updateTargetNet(self, episode):
        # No training done yet
        if not self.target.built: return
        if episode == 0: return
        self.target.set_weights(self.model.get_weights())

    def create_dqn_model(self):
        
        frames_input = tf.keras.layers.Input(shape=ATARI_SHAPE)

        denseLayer1 = tf.keras.layers.Dense(128,activation='relu')(frames_input)
        denseLayer2 = tf.keras.layers.Dense(256,activation='relu')(denseLayer1)
        outLayer = tf.keras.layers.Dense(ACTION_SIZE)(denseLayer2)
        
        initial_model = tf.keras.models.Model(inputs=frames_input,outputs=outLayer)

        initial_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        initial_model.compile(optimizer=initial_optimizer,loss=train_loss)

        return initial_model

    # Train model by memory batch
    def train_memory_batch(self, memory, model, log_dir):
        mini_batch = random.sample(memory, FLAGS.batch_size)
        history = np.zeros((FLAGS.batch_size, ATARI_SHAPE))
        next_history = np.zeros((FLAGS.batch_size, ATARI_SHAPE))
        target = np.zeros((FLAGS.batch_size,))
        action, reward = [], []

        for idx, val in enumerate(mini_batch):
            history[idx] = val[0]
            next_history[idx] = val[3]
            action.append(val[1])
            reward.append(val[2])
        
        # next_Q_values = model.predict(next_history)
        next_Q_values = self.target(np.array(next_history))

        for i in range(FLAGS.batch_size):
            target[i] = reward[i] + FLAGS.gamma * np.amax(next_Q_values[i])

        action_one_hot = get_one_hot(action, ACTION_SIZE)
        target_one_hot = action_one_hot * target[:, None]

        # ''''''
        # h = model.fit(
        #     history, target_one_hot, epochs=1,
        #     k

        with tf.GradientTape() as tape:
            predictions = self.model(np.array(history))
            predictions = tf.gather_nd(predictions, tf.stack((tf.range(FLAGS.batch_size), action), axis=1))
            loss = self.loss(target, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.trainLoss(loss)

        lossResult = self.trainLoss.result()

        return float(str("%.3f" % self.trainLoss.result()))

        # return h.history['loss'][0]

# Get action from model using epsilon 
def get_action(history, epsilon, step, model):
    if np.random.rand() <= epsilon or step <= FLAGS.observe_step_num:
        return random.randrange(ACTION_SIZE)
    else:
        q_value = model.predict(history.reshape(1,ATARI_SHAPE))
        return np.argmax(q_value[0])


# Store memory from memory replay
def store_memory(memory, history, action, reward, next_history, ):
    memory.append((history, action, reward, next_history))

def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]

def train():
    env = gym.make('LunarLander-v2')

    memory = deque(maxlen=FLAGS.replay_memory)
    episode_number = 0
    epsilon = FLAGS.init_epsilon
    epsilon_decay = (FLAGS.init_epsilon - FLAGS.final_epsilon) / FLAGS.epsilon_step_num
    global_step = 0
    startms = int(time.time() * 1000) 
    prev_step = 0

    moonLander = LandingBrain()

    if FLAGS.resume:
        model = load_model(FLAGS.restore_file_path)
        # Assume when we restore the model, the epsilon has already decreased to the final value
        epsilon = FLAGS.final_epsilon
    else:
        model = moonLander.create_dqn_model()

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    log_dir = "{}/run-{}-log".format(FLAGS.train_dir, now)
    # file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
    file_writer = tf.summary.create_file_writer(log_dir)

    start_timer = timer()

    while episode_number < FLAGS.num_episode:

        done = False
        dead = False
        step, score, start_life = 0, 0, 5
        loss = 0.0
        observe = env.reset()

        observe, _, _, _ = env.step(1)
        state = observe

        while not done:

            prior_state = observe

            # get action for the current history and go one step in environment
            action = get_action(prior_state, epsilon, global_step, model)

            # scale down epsilon, the epsilon only begin to decrease after observe steps
            if epsilon > FLAGS.final_epsilon and global_step > FLAGS.observe_step_num:
                epsilon -= epsilon_decay

            observe, reward, done, info = env.step(action)
            # pre-process the observation --> history
            next_state = observe

            store_memory(memory, prior_state, action, reward, next_state)  #

            # check if the memory is ready for training
            if global_step > FLAGS.observe_step_num:
                loss = loss + moonLander.train_memory_batch(memory, model, log_dir)
                # if loss > 100.0:
                #    print(loss)
                # if global_step % FLAGS.refresh_target_model_num == 0:  # update the target model
                #     model_target.set_weights(model.get_weights())
                # if global_step % 1000 and FLAGS.render:
                #     env.render()

            score += reward

            # If agent is dead, set the flag back to false, but keep the history unchanged,
            # to avoid to see the ball up in the sky

            # print("step: ", global_step)
            global_step += 1
            step += 1

            if done:
                if global_step <= FLAGS.observe_step_num:
                    state = "observe"
                elif FLAGS.observe_step_num < global_step <= FLAGS.observe_step_num + FLAGS.epsilon_step_num:
                    state = "explore"
                else:
                    state = "train"

                moonLander.updateTargetNet(episode_number)

                endms = int(time.time() * 1000) 
                total_ms_elapsed = endms - startms
                ms_per_step = (total_ms_elapsed) / float(global_step - prev_step)
                end_timer = timer()
                total_time_delta = timedelta(seconds=end_timer - start_timer)

                num_steps_per_episode = global_step / float(episode_number + 1)
                eta_left = (FLAGS.num_episode - episode_number) * (num_steps_per_episode) * (ms_per_step / 1000)

                startms = int(time.time() * 1000)
                prev_step = global_step

                print('state: {}, epsilon: {},episode: {}, score: {}, global_step: {}, avg loss: {}, step: {}, memory length: {}, ms_per_step: {:.2f}, elapsed_time: {}, eta: {}'
                      .format(state, round(epsilon,2), episode_number, score, global_step, loss / float(step), step, len(memory), ms_per_step, total_time_delta, time.strftime("%d:%H:%M:%S",time.gmtime(eta_left))))


                if episode_number % 100 == 0 or (episode_number + 1) == FLAGS.num_episode:
                #if episode_number % 1 == 0 or (episode_number + 1) == FLAGS.num_episode:  # debug
                    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
                    file_name = "latest_breakout_model.h5"
                    model_path = os.path.join(FLAGS.train_dir, file_name)
                    model.save(model_path)

                with file_writer.as_default():
                    # Add user custom data to TensorBoard
                    # loss_summary = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=loss / float(step))])
                    tf.summary.scalar("loss",loss / float(step), step=episode_number)

                    # score_summary = tf.Summary(value=[tf.Summary.Value(tag="score", simple_value=score)])
                    # file_writer.add_summary(score_summary, global_step=episode_number)
                    tf.summary.scalar("score_summary", score ,step=episode_number)
                    file_writer.flush()

                episode_number += 1

    file_writer.close()


def test():
    env = gym.make('LunarLander-v2')

    episode_number = 0
    epsilon = 0.001
    global_step = FLAGS.observe_step_num+1
    # model = load_model(FLAGS.restore_file_path)
    model = load_model(FLAGS.restore_file_path)  # load model with customized loss func

    # test how to deep copy a model
    '''
    model_copy = clone_model(model)    # only copy the structure, not the value of the weights
    model_copy.set_weights(model.get_weights())
    '''

    while episode_number < FLAGS.num_episode:

        done = False
        dead = False
        # 1 episode = 5 lives
        score, start_life = 0, 5
        observe = env.reset()

        observe, _, _, _ = env.step(1)
        # At start of episode, there is no preceding frame
        # So just copy initial states to make history
        state = observe
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:
            env.render()
            #time.sleep(0.01)

            # get action for the current history and go one step in environment
            action = get_action(history, epsilon, global_step, model)
            # change action to real_action
            real_action = action + 1

            observe, reward, done, info = env.step(real_action)
            next_state = observe
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            # if the agent missed ball, agent is dead --> episode is not over
            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            # TODO: may be we should give negative reward if miss ball (dead)
            reward = np.clip(reward, -1., 1.)

            score += reward

            # If agent is dead, set the flag back to false, but keep the history unchanged,
            # to avoid to see the ball up in the sky
            if dead:
                dead = False
            else:
                history = next_history

            # print("step: ", global_step)
            global_step += 1

            if done:
                episode_number += 1
                print('episode: {}, score: {}'.format(episode_number, score))


def main(argv=None):
    train()
    # test()


if __name__ == '__main__':
    main()
