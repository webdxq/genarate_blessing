
import tensorflow as tf  

def neural_network(input_data, word_size, batch_size, model='lstm', rnn_size=128, num_layers=2):  
    if model == 'rnn':  
        cell_fun = tf.nn.rnn_cell.BasicRNNCell  
    elif model == 'gru':  
        cell_fun = tf.nn.rnn_cell.GRUCell  
    elif model == 'lstm':  
        cell_fun = tf.nn.rnn_cell.BasicLSTMCell  
   
    cell = cell_fun(rnn_size, state_is_tuple=True)  
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)  
   
    initial_state = cell.zero_state(batch_size, tf.float32)  
    with tf.variable_scope('rnnlm',reuse=tf.AUTO_REUSE):  
        softmax_w = tf.get_variable("softmax_w", [rnn_size, word_size])  
        softmax_b = tf.get_variable("softmax_b", [word_size])  
        with tf.device("/cpu:0"):  
            embedding = tf.get_variable("embedding", [word_size, rnn_size])  
            inputs = tf.nn.embedding_lookup(embedding, input_data)  

    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')  
    output = tf.reshape(outputs,[-1, rnn_size])  
   
    logits = tf.matmul(output, softmax_w) + softmax_b  
    probs = tf.nn.softmax(logits)  
    return logits, last_state, probs, cell, initial_state 

# def inference_step(self, sess, probs, last_state, input_feed, state_feed):
#     softmax_output, state_output = sess.run(
#         [probs, last_state],
#         feed_dict={
#             "input_feed:0": input_feed,
#             "lstm/state_feed:0": state_feed,
#         })
#     return softmax_output, state_output