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
        self.restore_file_path = './tf_train_breakout/latest_breakout_model-2.h5'
        self.num_episode = 100000
        self.observe_step_num = 50000
        self.epsilon_step_num = 1000000
        self.refresh_target_model_num = 10000
        # 46000 is 3.5GB 
        self.replay_memory = 184000
        self.no_op_steps = 30
        self.regularizer_scale = 0.01
        self.batch_size = 32
        self.learning_rate = 0.00025
        self. init_epsilon = 1.0
        self.final_epsilon = 0.1
        self.gamma = 0.99
        self.resume = False
        self.render = False

FLAGS = ParametersClass()

ATARI_SHAPE = (84, 84, 4)  
ACTION_SIZE = 3

# Process frames to 84
def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe


def huber_loss(y, q_value):
    error = K.abs(y - q_value)
    quadratic_part = K.clip(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
    return loss

def create_atari_model():

    frames_input = tf.keras.layers.Input(shape=ATARI_SHAPE)
    actions_input = layers.Input((ACTION_SIZE,))

    lambdaLayer = tf.keras.layers.Lambda(lambda x: x / 255.0)(frames_input)
    conv1Layer = tf.keras.layers.Conv2D(16,[8,8],strides=(4,4),activation='relu')(lambdaLayer)
    conv2Layer = tf.keras.layers.Conv2D(32,[4,4],strides=(2,2),activation='relu')(conv1Layer)
    flattenLayer = tf.keras.layers.Flatten()(conv2Layer)
    dense1Layer = tf.keras.layers.Dense(256,activation='relu')(flattenLayer)
    dense2Layer = tf.keras.layers.Dense(ACTION_SIZE)(dense1Layer)
    multiplyLayer = tf.keras.layers.multiply([dense2Layer, actions_input])
    
    initial_optimizer=RMSprop(lr=FLAGS.learning_rate, rho=0.95, epsilon=0.01)

    initial_model = tf.keras.models.Model(inputs=[frames_input,actions_input],outputs=multiplyLayer)

    # initial_model.build()
    initial_model.compile(optimizer=initial_optimizer,loss=huber_loss)

    return initial_model

def create_dqn_model():

    frames_input = tf.keras.layers.Input(shape=ATARI_SHAPE)
    actions_input = layers.Input((ACTION_SIZE,))

    lambdaLayer = tf.keras.layers.Lambda(lambda x: x / 255.0)(frames_input)
    conv1Layer = tf.keras.layers.Conv2D(32,[8,8],strides=(4,4),activation='relu')(lambdaLayer)
    conv2Layer = tf.keras.layers.Conv2D(64,[4,4],strides=(2,2),activation='relu')(conv1Layer)
    conv3Layer = tf.keras.layers.Conv2D(64,[3,3],activation='relu')(conv2Layer)
    flattenLayer = tf.keras.layers.Flatten()(conv3Layer)

    dense1Layer = tf.keras.layers.Dense(512,activation='relu')(flattenLayer)

    valueLayer = tf.keras.layers.Dense(512,activation='relu')(dense1Layer)
    valueOutLayer = tf.keras.layers.Dense(1,activation='relu')(valueLayer)

    advantageLayer = tf.keras.layers.Dense(512,activation='relu')(dense1Layer)
    advantageOutLayer = tf.keras.layers.Dense(ACTION_SIZE, activation='relu')(advantageLayer)
    normalizedAdvantage = tf.keras.layers.Lambda(lambda x: x - tf.reduce_mean(x))(advantageOutLayer)

    qLayer = tf.keras.layers.Add()([valueOutLayer,normalizedAdvantage])

    multiplyLayer = tf.keras.layers.multiply([qLayer, actions_input])
    
    initial_optimizer=RMSprop(lr=FLAGS.learning_rate, rho=0.95, epsilon=0.01)

    initial_model = tf.keras.models.Model(inputs=[frames_input,actions_input],outputs=multiplyLayer)

    # initial_model.build()
    initial_model.compile(optimizer=initial_optimizer,loss=huber_loss)

    return initial_model

# Get action from model using epsilon 
def get_action(history, epsilon, step, model):
    if np.random.rand() <= epsilon or step <= FLAGS.observe_step_num:
        return random.randrange(ACTION_SIZE)
    else:
        q_value = model.predict([history, np.ones(ACTION_SIZE).reshape(1, ACTION_SIZE)])
        return np.argmax(q_value[0])


# Store memory from memory replay
def store_memory(memory, history, action, reward, next_history, dead):
    memory.append((history, action, reward, next_history, dead))

def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]

# Train model by memory batch
def train_memory_batch(memory, model, log_dir):
    mini_batch = random.sample(memory, FLAGS.batch_size)
    history = np.zeros((FLAGS.batch_size, ATARI_SHAPE[0],
                        ATARI_SHAPE[1], ATARI_SHAPE[2]))
    next_history = np.zeros((FLAGS.batch_size, ATARI_SHAPE[0],
                             ATARI_SHAPE[1], ATARI_SHAPE[2]))
    target = np.zeros((FLAGS.batch_size,))
    action, reward, dead = [], [], []

    for idx, val in enumerate(mini_batch):
        history[idx] = val[0]
        next_history[idx] = val[3]
        action.append(val[1])
        reward.append(val[2])
        dead.append(val[4])

    actions_mask = np.ones((FLAGS.batch_size, ACTION_SIZE))
    next_Q_values = model.predict([next_history, actions_mask])

    for i in range(FLAGS.batch_size):
        if dead[i]:
            target[i] = -1
            # target[i] = reward[i]
        else:
            target[i] = reward[i] + FLAGS.gamma * np.amax(next_Q_values[i])

    action_one_hot = get_one_hot(action, ACTION_SIZE)
    target_one_hot = action_one_hot * target[:, None]

    # ''''''
    h = model.fit(
        [history, action_one_hot], target_one_hot, epochs=1,
        batch_size=FLAGS.batch_size, verbose=0)

    return h.history['loss'][0]


def train():
    env = gym.make('BreakoutDeterministic-v4')

    memory = deque(maxlen=FLAGS.replay_memory)
    episode_number = 0
    epsilon = FLAGS.init_epsilon
    epsilon_decay = (FLAGS.init_epsilon - FLAGS.final_epsilon) / FLAGS.epsilon_step_num
    global_step = 0
    startms = int(time.time() * 1000) 
    prev_step = 0

    if FLAGS.resume:
        model = load_model(FLAGS.restore_file_path)
        # Assume when we restore the model, the epsilon has already decreased to the final value
        epsilon = FLAGS.final_epsilon
    else:
        # model = create_atari_model()
        model = create_dqn_model()

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    log_dir = "{}/run-{}-log".format(FLAGS.train_dir, now)
    # file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
    file_writer = tf.summary.create_file_writer(log_dir)

    model_target = clone_model(model)
    model_target.set_weights(model.get_weights())

    start_timer = timer()

    while episode_number < FLAGS.num_episode:

        done = False
        dead = False
        step, score, start_life = 0, 0, 5
        loss = 0.0
        observe = env.reset()

        # Wait at beginning
        for _ in range(random.randint(1, FLAGS.no_op_steps)):
            observe, _, _, _ = env.step(1)
        # Copy frames over to lay out memory
        state = pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:
            if FLAGS.render:
                env.render()
                time.sleep(0.01)

            # get action for the current history and go one step in environment
            action = get_action(history, epsilon, global_step, model_target)
            # change action to real_action
            real_action = action + 1

            # scale down epsilon, the epsilon only begin to decrease after observe steps
            if epsilon > FLAGS.final_epsilon and global_step > FLAGS.observe_step_num:
                epsilon -= epsilon_decay

            observe, reward, done, info = env.step(real_action)
            # pre-process the observation --> history
            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            # if the agent missed ball, agent is dead --> episode is not over
            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            # TODO: may be we should give negative reward if miss ball (dead)
            # reward = np.clip(reward, -1., 1.)  # clip here is not correct

            # save the statue to memory, each replay takes 2 * (84*84*4) bytes = 56448 B = 55.125 KB
            store_memory(memory, history, action, reward, next_history, dead)  #

            # check if the memory is ready for training
            if global_step > FLAGS.observe_step_num:
                loss = loss + train_memory_batch(memory, model, log_dir)
                # if loss > 100.0:
                #    print(loss)
                if global_step % FLAGS.refresh_target_model_num == 0:  # update the target model
                    model_target.set_weights(model.get_weights())

            score += reward

            # If agent is dead, set the flag back to false, but keep the history unchanged,
            # to avoid to see the ball up in the sky
            if dead:
                dead = False
            else:
                history = next_history

            #print("step: ", global_step)
            global_step += 1
            step += 1

            if done:
                if global_step <= FLAGS.observe_step_num:
                    state = "observe"
                elif FLAGS.observe_step_num < global_step <= FLAGS.observe_step_num + FLAGS.epsilon_step_num:
                    state = "explore"
                else:
                    state = "train"

                endms = int(time.time() * 1000) 
                total_ms_elapsed = endms - startms
                ms_per_step = (total_ms_elapsed) / float(global_step - prev_step)
                end_timer = timer()
                total_time_delta = timedelta(seconds=end_timer - start_timer)

                num_steps_per_episode = global_step / float(episode_number + 1)
                eta_left = (FLAGS.num_episode - episode_number) * (num_steps_per_episode) * (ms_per_step / 1000)

                startms = int(time.time() * 1000)
                prev_step = global_step

                print('state: {}, episode: {}, score: {}, global_step: {}, avg loss: {}, step: {}, memory length: {}, ms_per_step: {:.2f}, elapsed_time: {}, eta: {}'
                      .format(state, episode_number, score, global_step, loss / float(step), step, len(memory), ms_per_step, total_time_delta, time.strftime("%H:%M:%S",time.gmtime(eta_left))))


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
    env = gym.make('BreakoutDeterministic-v4')

    episode_number = 0
    epsilon = 0.001
    global_step = FLAGS.observe_step_num+1
    # model = load_model(FLAGS.restore_file_path)
    model = load_model(FLAGS.restore_file_path, custom_objects={'huber_loss': huber_loss})  # load model with customized loss func

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
        state = pre_processing(observe)
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
            # pre-process the observation --> history
            next_state = pre_processing(observe)
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
    # train()
    test()


if __name__ == '__main__':
    main()
