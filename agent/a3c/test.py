import os
import tensorflow as tf
import numpy as np

from agent.a3c.network import AC_Network
from environment.assembly import Assembly
from environment.panelblock import *


if __name__ == '__main__':
    panel_blocks, num_of_processes = import_panel_block_schedule('../../environment/data/PBS_assy_sequence_gen_000.csv')

    len_of_queue = 10
    s_size = num_of_processes * len_of_queue + num_of_processes
    a_size = len_of_queue

    model_path = '../../model/a3c/queue-%d' % a_size
    test_path = '../../test/a3c/queue-%d' % a_size

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    env = Assembly(num_of_processes, len_of_queue, inbound_panel_blocks=panel_blocks)

    tf.reset_default_graph()
    with tf.Session() as sess:
        network = AC_Network(s_size, a_size, 'global', None)
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver = tf.train.Saver(max_to_keep=5)
        saver.restore(sess, ckpt.model_checkpoint_path)

        s = env.reset()

        while True:
            a_dist, v = sess.run([network.policy, network.value], feed_dict={network.inputs: [s]})
            a = np.argmax(a_dist[0])

            s1, r, d = env.step(a)

            if d:
                print("total_lead time: {0}".format(env.model['Sink'].last_arrival))
                break

            s = s1