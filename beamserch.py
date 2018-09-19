#-*-coding:utf-8-*-
import tensorflow as tf

def rnn_beam_search(update_fn, initial_state, sequence_length, beam_width,
                    begin_token_id, end_token_id, name="rnn"):
  """Beam-search decoder for recurrent models.

  Args:
    update_fn: Function to compute the next state and logits given the current
               state and ids.
    initial_state: Recurrent model states.
    sequence_length: Length of the generated sequence.
    beam_width: Beam width.
    begin_token_id: Begin token id.
    end_token_id: End token id.
    name: Scope of the variables.
  Returns:
    ids: Output indices.
    logprobs: Output log probabilities probabilities.
  """
  # batch_size = initial_state.shape.as_list()[0]
  # 
  batch_size = initial_state[0][0].get_shape()[0]
  #change state frame to (batch_size, )
  state = tf.tile(tf.expand_dims(initial_state, axis=1), [1, beam_width, 1])

  # +是concate操作，合并两个列表，tf.log对两个列表里的元素依次应用log函数
  # 此处得出值是[[  0. -inf -inf]] ，beam_width是-inf, (1,1+beam_width)
  sel_sum_logprobs = tf.log([[1.] + [0.] * (beam_width - 1)])
  #初始输入 (batch_size, beam_width)
  ids = tf.tile([[begin_token_id]], [batch_size, beam_width])
  #选择的id
  sel_ids = tf.zeros([batch_size, beam_width, 0], dtype=ids.dtype)
  #掩码
  mask = tf.ones([batch_size, beam_width], dtype=tf.float32)

  for i in range(sequence_length):
    with tf.variable_scope(name, reuse=True if i > 0 else None):

      state, logits = update_fn(state, ids)
      #tf.nn.log_softmax() :input/output(batch_size, num_size)
      logits = tf.nn.log_softmax(logits)
      '''
      [[[  0.]
        [-inf]
        [-inf]]]
        tf.expand_dims(sel_sum_logprobs, axis=2) : (1,beam_width+1,1)
        tf.expand_dims(mask, axis=2) : (batch_size, beam_width, 1)
        logits * tf.expand_dims(mask, axis=2) : 
      '''
      sum_logprobs = (
          tf.expand_dims(sel_sum_logprobs, axis=2) +
          (logits * tf.expand_dims(mask, axis=2)))

      num_classes = logits.shape.as_list()[-1]

      sel_sum_logprobs, indices = tf.nn.top_k(
          tf.reshape(sum_logprobs, [batch_size, num_classes * beam_width]),
          k=beam_width)

      ids = indices % num_classes

      beam_ids = indices // num_classes

      state = batch_gather(state, beam_ids)

      sel_ids = tf.concat([batch_gather(sel_ids, beam_ids),
                           tf.expand_dims(ids, axis=2)], axis=2)

      mask = (batch_gather(mask, beam_ids) *
              tf.to_float(tf.not_equal(ids, end_token_id)))

  return sel_ids, sel_sum_logprobs

def batch_gather(tensor, indices):
  """Gather in batch from a tensor of arbitrary size.

  In pseudocode this module will produce the following:
  output[i] = tf.gather(tensor[i], indices[i])

  Args:
    tensor: Tensor of arbitrary size.
    indices: Vector of indices.
  Returns:
    output: A tensor of gathered values.
  """
  shape = get_shape(tensor)
  flat_first = tf.reshape(tensor, [shape[0] * shape[1]] + shape[2:])
  indices = tf.convert_to_tensor(indices)
  offset_shape = [shape[0]] + [1] * (indices.shape.ndims - 1)
  offset = tf.reshape(tf.range(shape[0]) * shape[1], offset_shape)
  output = tf.gather(flat_first, indices + offset)
  return output