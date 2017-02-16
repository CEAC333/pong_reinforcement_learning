import tensorflow as tf
import cv2  # read in pixel data
import pong  # our class
import numpy as np  # math
import random  # random
# queue data structure. fast appends. and pops. replay memory
from collections import deque


# hyper params
ACTIONS = 3  # up,down, stay
# define our learning rate
GAMMA = 0.99
# for updating our gradient or training over time
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
# how many frames to anneal epsilon
EXPLORE = 500000
OBSERVE = 50000
# store our experiences, the size of it
REPLAY_MEMORY = 500000
# batch size to train on
BATCH = 100
# input image size in pixels
INPUT_SIZE = 84

# create tensorflow graph
def createGraph():

    # CNN
    # creates an empty tensor with all elements set to zero with a shape

    # YOUR CODE HERE

    # input for pixel data
    s = tf.placeholder("float", [None, INPUT_SIZE, INPUT_SIZE, 4])

    # Computes rectified linear unit activation fucntion on  a 2-D convolution
    # given 4-D input and filter tensors.
    
    # YOUR CODE HERE

    # return input and output to the network
    return s, fc5


# deep q network. feed in pixel data to graph session
def trainGraph(inp, out, sess):

    # to calculate the argmax, we multiply
    # the predicted output with a vector
    # with one value 1 and rest as 0
    argmax = tf.placeholder("float", [None, ACTIONS])
    gt = tf.placeholder("float", [None])  # ground truth

    # action

    # YOUR CODE HERE

    # cost function we will reduce through backpropagation
    
    # YOUR CODE HERE

    # optimization function to reduce our minimize our cost function
    
    # YOUR CODE HERE


    # initialize our game
    game = pong.PongGame()

    # create a queue for experience replay to store policies
    D = deque()

    # intial frame
    frame = game.getPresentFrame()
    # convert rgb to gray scale for processing
    frame = cv2.cvtColor(cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE)), cv2.COLOR_BGR2GRAY)
    # binary colors, black or white
    ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    # stack frames, that is our input tensor
    inp_t = np.stack((frame, frame, frame, frame), axis=2)

    # saver
    saver = tf.train.Saver()

    sess.run(tf.initialize_all_variables())

    t = 0
    epsilon = INITIAL_EPSILON

    # training time
    while(1):
        # output tensor
        out_t = out.eval(feed_dict={inp: [inp_t]})[0]
        # argmax function
        argmax_t = np.zeros([ACTIONS])

        # random action with prob epsilon
        if(random.random() <= epsilon):
            maxIndex = random.randrange(ACTIONS)
        # predicted action with prob (1 - epsilon)
        else:
            maxIndex = np.argmax(out_t)
        argmax_t[maxIndex] = 1

        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # reward tensor if score is positive
        reward_t, frame = game.getNextFrame(argmax_t)

        # get frame pixel data
        frame = cv2.cvtColor(cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE)), cv2.COLOR_BGR2GRAY)
        ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        frame = np.reshape(frame, (INPUT_SIZE, INPUT_SIZE, 1))

        # new input tensor
        inp_t1 = np.append(frame, inp_t[:, :, 0:3], axis=2)

        # add our input tensor, argmax tensor, reward and updated input tensor
        # to stack of experiences
        
        # YOUR CODE HERE

        # if we run out of replay memory, make room
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # training iteration
        if t > OBSERVE:

            # get values from our replay memory
            minibatch = random.sample(D, BATCH)

            inp_batch = [d[0] for d in minibatch]
            argmax_batch = [d[1] for d in minibatch]
            reward_batch = [d[2] for d in minibatch]
            inp_t1_batch = [d[3] for d in minibatch]

            gt_batch = []
            out_batch = out.eval(feed_dict={inp: inp_t1_batch})

            # add values to our batch
            for i in range(0, len(minibatch)):
                gt_batch.append(reward_batch[i] + GAMMA * np.max(out_batch[i]))

            # train on that
            train_step.run(feed_dict={
                           gt: gt_batch,
                           argmax: argmax_batch,
                           inp: inp_batch
                           })

        # update our input tensor the the next frame
        inp_t = inp_t1
        t = t+1

        # print our where wer are after saving where we are
        if t % 10000 == 0:
            saver.save(sess, './' + 'pong' + '-dqn', global_step=t)

        print("TIMESTEP", t, "/ EPSILON", epsilon, "/ ACTION", maxIndex,
              "/ REWARD", reward_t, "/ Q_MAX %e" % np.max(out_t))


def main():
    # create session
    sess = tf.InteractiveSession()
    # input layer and output layer by creating graph
    inp, out = createGraph()
    # train our graph on input and output with session variables
    trainGraph(inp, out, sess)

if __name__ == "__main__":
    main()
