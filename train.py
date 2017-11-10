##########################
#### Read Config File #### {{{

import sys, time
start_time = time.time()
if len(sys.argv) < 3:
  print "EXPECT: "+sys.argv[0]+" {train|extract} CONFIGFILE [DEPENDENCYGRAPHFILE]"
  exit()
if sys.argv[1] not in [ "train", "extract"]:
  print "EXPECT: "+sys.argv[0]+" {train|extract} CONFIGFILE [DEPENDENCYGRAPHFILE]"
  exit()

mode = sys.argv[1]
configfile = sys.argv[2]

print_info = mode == "train"

def info(msg):
  if print_info:
    print "INFO: "+msg

info("Load Configuration...")

import ConfigParser
config = ConfigParser.RawConfigParser()
config.read(configfile)

embeddings_file_regular_words = config.get('Data', 'embeddings_file_regular_words')
embeddings_file_postags_deplabels = config.get('Data', 'embeddings_file_postags_deplabels')
trainingdata_file = config.get('Data', 'trainingdata_file')

number_of_position_embedded_words_per_DG_node = 2
number_of_regular_words_embeddings_per_DG_node = config.getint('Model', 'number_of_regular_words_embeddings_per_DG_node')
number_of_postags_deplabels_embeddings_per_DG_node = config.getint('Model', 'number_of_postags_deplabels_embeddings_per_DG_node')
number_of_lstm_layers = config.getint('Model', 'number_of_lstm_layers')
number_of_lstm_units_per_layer = config.getint('Model', 'number_of_lstm_units_per_layer')
max_number_of_nodes_per_DG = config.getint('Model', 'max_number_of_nodes_per_DG')
position_embedding_size = config.getint('Model', 'position_embedding_size')
distinct_fw_and_bw_encoders = config.getboolean('Model', 'distinct_fw_and_bw_encoders')

batch_size = config.getint('Training', 'batch_size')
epochs = config.getint('Training', 'epochs')
batches_per_epoch = config.getint('Training', 'batches_per_epoch')
test_batches = config.getint('Training', 'test_batches')
validation_batches = config.getint('Training', 'validation_batches')
batches_per_bucket = config.getint('Training', 'batches_per_bucket')
validation_stop = config.getboolean('Training', 'validation_stop')
validation_increase_tolerate = config.getint('Training', 'validation_increase_tolerate')
session_name = config.get('Training', 'session_name')
save_session = config.getboolean('Training', 'save_session')
continue_session = config.getboolean('Training', 'continue_session')
use_GRU = config.getboolean('Training', 'use_GRU')
dependency_graphs_file = None
max_prediction_length = None

if mode == "extract":
  dependency_graphs_file = config.get('Data', 'dependency_graphs_file') # expect conll-X (2006) format
  if len(sys.argv) == 4:
    dependency_graphs_file = sys.argv[3]
  max_prediction_length = config.getint('Model', 'max_prediction_length')
  batch_size = 1
  epochs = None
  batches_per_epoch = None
  test_batches = None
  validation_batches = None
  batches_per_bucket = None
  validation_stop = None
  validation_increase_tolerate = None
  save_session = None
  continue_session = None
  trainingdata_file = None

info("DONE") # }}}


##################################
#### Load Embeddings and Data #### {{{
info("Load embeddings and data...")

import pickle
import numpy
from sklearn.cross_validation import train_test_split

embedding_matrix_regular_words = None
embedding_matrix_postags_deplabels = None
word_to_index_regular_words = None
word_to_index_postags_deplabels = None

X = None
Y = None

with open(embeddings_file_regular_words) as embeddings_file:
    (embedding_matrix_regular_words, word_to_index_regular_words, number_of_placeholder_words) = pickle.load(embeddings_file)

with open(embeddings_file_postags_deplabels) as embeddings_file:
    (embedding_matrix_postags_deplabels, word_to_index_postags_deplabels) = pickle.load(embeddings_file)

number_of_words_per_DG_node = number_of_position_embedded_words_per_DG_node + number_of_regular_words_embeddings_per_DG_node + number_of_postags_deplabels_embeddings_per_DG_node

if mode == "train":
  with open(trainingdata_file) as t_file:
    (X, Y) = pickle.load(t_file)

  info("data samples: "+ str(len(X)))
  info("remove samples with more than " + str(max_number_of_nodes_per_DG) + " nodes")
  i = 0
  while i < len(X):
    if len(X[i]) > max_number_of_nodes_per_DG:
      del X[i]
      del Y[i]
    else:
      i = i + 1
  info("data samples: "+ str(len(X)))

  # DG : abbreviation for dependency graph
  assert number_of_words_per_DG_node == len(X[0][0])

  X, X_test, Y, Y_test = train_test_split(X, Y, test_size=test_batches*batch_size, random_state=123)
  X, X_validate, Y, Y_validate = train_test_split(X, Y, test_size=validation_batches*batch_size, random_state=123)

  def split_into_buckets(X, Y, batch_size, batches_per_bucket):
    assert len(X) == len(Y)
    bucketsize = batch_size * batches_per_bucket
    nbuckets = len(X)/bucketsize
    x_y_tuples = [ (X[i],Y[i]) for i in range(len(X)) ]
    x_y_tuples = sorted(x_y_tuples, key=lambda a: len(a[1]))
    buckets = []

    for i in range(nbuckets):
        buckets.append(x_y_tuples[i*bucketsize:(i+1)*bucketsize])
    buckets[-1] += x_y_tuples[nbuckets*bucketsize:] # add remaining buckets to last bucket -> last bucket will be larger than the others
    X_buckets = [[ x[0] for x in bucket] for bucket in buckets]
    Y_buckets = [[ y[1] for y in bucket] for bucket in buckets]
    return X_buckets, Y_buckets

  X_buckets, Y_buckets = split_into_buckets(X, Y, batch_size, batches_per_bucket)

  del X
  del Y

  info("test samples: "+ str(len(X_test)))
  info("validation samples: "+ str(len(X_validate)))

  info("DONE") #}}}

###############################
#### Construct Model Graph ####

# set a few params {{{
embedding_dimensions_regular_words = embedding_matrix_regular_words.shape[1]
embedding_dimensions_postags_deplabels = embedding_matrix_postags_deplabels.shape[1]
embedding_shape_ids = [max_number_of_nodes_per_DG + 1, position_embedding_size]
vocabulary_size = len(word_to_index_regular_words)

import tensorflow as tf

x = tf.placeholder(tf.int32, shape=[None, batch_size, number_of_words_per_DG_node])
y = tf.placeholder(tf.int32, shape=[None, batch_size])

i_regular_words_start = number_of_position_embedded_words_per_DG_node
i_regular_words_end   = number_of_position_embedded_words_per_DG_node + number_of_regular_words_embeddings_per_DG_node

#}}}

### Embedding Layer ### {{{
# structure of each node is node_id, head_id, form, lemma, postag, deprel

info("Set up embedding layer...")
with tf.name_scope('embedding') as scope:

    x_node_id = x[:,:, 0 ]
    x_head_id = x[:,:, 1 ]
    x_regular_words_embedded     = x[:,:, i_regular_words_start : i_regular_words_end ]
    x_postags_deplabels_embedded = x[:,:, i_regular_words_end   : ]

    W_embedding_layer_regular_words = tf.Variable(embedding_matrix_regular_words, name='embedding_matrix_regular_words')
    W_embedding_layer_postags_deplabels = tf.Variable(embedding_matrix_postags_deplabels, name='embedding_matrix_postags_deplabels')
    W_embedding_layer_node_id = tf.Variable(tf.random_normal( embedding_shape_ids, mean = 1, stddev = 0.1, dtype=tf.float32, seed = 123), name= 'embedding_node_id' )
    W_embedding_layer_head_id = tf.Variable(tf.random_normal( embedding_shape_ids, mean = 1, stddev = 0.1, dtype=tf.float32, seed = 123), name= 'embedding_head_id' )

    embedding_lookup_regular_words = tf.nn.embedding_lookup(W_embedding_layer_regular_words, x_regular_words_embedded, name='embedding_lookup_regular_words')
    embedding_lookup_postags_deplabels = tf.nn.embedding_lookup(W_embedding_layer_postags_deplabels, x_postags_deplabels_embedded, name='embedding_lookup_postags_deplabels')
    embedding_lookup_node_id = tf.nn.embedding_lookup(W_embedding_layer_node_id, x_node_id, name='embedding_lookup_node_id')
    embedding_lookup_head_id = tf.nn.embedding_lookup(W_embedding_layer_head_id, x_head_id, name='embedding_lookup_head_id')

    embedding_lookup_regular_words_float = tf.cast(embedding_lookup_regular_words, tf.float32)
    embedding_lookup_postags_deplabels_float = tf.cast(embedding_lookup_postags_deplabels, tf.float32)

    embedding_lookup_regular_words_float_reshape = \
      tf.reshape(embedding_lookup_regular_words_float, \
      [-1, batch_size, number_of_regular_words_embeddings_per_DG_node * embedding_dimensions_regular_words])
    embedding_lookup_postags_deplabels_float_reshape = \
      tf.reshape(embedding_lookup_postags_deplabels_float, \
      [-1, batch_size, number_of_postags_deplabels_embeddings_per_DG_node * embedding_dimensions_postags_deplabels])


info("DONE")


info("Combine embeddings")
with tf.name_scope('combine_input') as scope:
    recombined_DG = tf.concat((
      embedding_lookup_node_id,
      embedding_lookup_head_id,
      embedding_lookup_regular_words_float_reshape,
      embedding_lookup_postags_deplabels_float_reshape
      ), 2, name = "recombined_DG")
info("DONE") #}}}

### Encoder ### {{{

dropout_output_keep_prob = tf.placeholder(tf.float32)

info("Set up encoder...")
def lstm_cell(number_of_units):
    return tf.contrib.rnn.LSTMCell(number_of_units)

def gru_cell(number_of_units):
    return tf.contrib.rnn.GRUCell(number_of_units, activation=tf.nn.relu)

def use_cell(number_of_units):
    if use_GRU:
      ret = gru_cell(number_of_units)
    else:
      ret = lstm_cell(number_of_units)
    if mode == "train":
      return tf.contrib.rnn.DropoutWrapper(ret, output_keep_prob=dropout_output_keep_prob, seed=123)
    else:
      return ret

with tf.name_scope('Encoder') as scope:
    encoder_stacked_rnn_fw = tf.contrib.rnn.MultiRNNCell([
        use_cell(number_of_lstm_units_per_layer/2) for _ in range(number_of_lstm_layers)])

    encoder_stacked_rnn_bw = encoder_stacked_rnn_fw
    if distinct_fw_and_bw_encoders:
       encoder_stacked_rnn_bw = tf.contrib.rnn.MultiRNNCell([
          use_cell(number_of_lstm_units_per_layer/2) for _ in range(number_of_lstm_layers)])

    encoder_outputs, (encoder_final_state_fw, encoder_final_state_bw) = tf.nn.bidirectional_dynamic_rnn(
        encoder_stacked_rnn_fw, encoder_stacked_rnn_bw, recombined_DG, dtype=tf.float32,
        time_major = True, scope='Encoder')
    del encoder_outputs

    encoder_final_state = ( tf.concat( (encoder_final_state_fw[0], encoder_final_state_bw[0]), 1, name = 'encoder_final_state_0'),
                            tf.concat( (encoder_final_state_fw[1], encoder_final_state_bw[1]), 1, name = 'encoder_final_state_1'),
                            tf.concat( (encoder_final_state_fw[2], encoder_final_state_bw[2]), 1, name = 'encoder_final_state_2') )
info("DONE") # }}}

### Decoder ### {{{

info("Set up decoder...")


with tf.name_scope('fully_connected') as scope:
    W = tf.Variable(tf.zeros([number_of_lstm_units_per_layer, vocabulary_size]), dtype=tf.float32, name='weights')
    b = tf.Variable(tf.zeros([vocabulary_size]), dtype=tf.float32, name = 'biases')

with tf.name_scope('Decoder') as scope:

    decoder_input_ground_truth = tf.placeholder(shape=(None, batch_size), dtype=tf.int32, name='decoder_input')
    decoder_input_ground_truth_embedded = tf.nn.embedding_lookup(W_embedding_layer_regular_words, decoder_input_ground_truth, name='decoder_input_embedding_lookup')

    initial_state = encoder_final_state # this is the encoded information

    decoder_stacked_rnn = tf.contrib.rnn.MultiRNNCell([
        use_cell(number_of_lstm_units_per_layer) for _ in range(number_of_lstm_layers)])

    feed_prediction_prob = tf.placeholder(tf.float32, shape=(1), name='feed_prediction_prob')

    initial_decoder_input = tf.nn.embedding_lookup(W_embedding_layer_regular_words,[word_to_index_regular_words['<empty>'] for i in range(batch_size)], name='decoder_input_embedding_lookup')
    initial_finished = tf.constant([False for i in range(batch_size)], dtype=tf.bool)
    initial_finished_t = tf.constant([True for i in range(batch_size)], dtype=tf.bool)
    eos_const = tf.constant(word_to_index_regular_words['<eos>'], dtype=tf.int32)
    empty_const = tf.constant(word_to_index_regular_words['<empty>'], dtype=tf.int32)
    max_time_const = None
    if mode == "extract":
      max_time_const = tf.constant(max_prediction_length, dtype=tf.int32)

    def raw_rnn_loop_fn_train(time, cell_output, cell_state, loop_state):

      if cell_output is None:
        finished = initial_finished
        next_input = initial_decoder_input
        next_cell_state = initial_state
        emit_output = cell_output
        return (finished, next_input, next_cell_state, emit_output, 0)

      else:
        finished_ground_truth_at_eos = tf.equal(decoder_input_ground_truth[loop_state], eos_const)
        finished = finished_ground_truth_at_eos
        prediction = tf.argmax(tf.add(tf.matmul(cell_output,W), b), axis=1, output_type=tf.int32)
        prediction_embedded = tf.nn.embedding_lookup(W_embedding_layer_regular_words, prediction, name='decoder_input_embedding_lookup')
        ground_truth = decoder_input_ground_truth_embedded[loop_state]
        roulette = tf.random_uniform(feed_prediction_prob.shape, minval = 0, maxval = 1, dtype=tf.float32, name="feed_prediction_to_decoder_roulette")
        next_input = tf.cond( (roulette <= feed_prediction_prob)[0] , lambda: prediction_embedded, lambda: ground_truth)
        next_cell_state = cell_state
        emit_output = cell_output
        return (finished, next_input, next_cell_state, emit_output, loop_state + 1)

    def raw_rnn_loop_fn_extract(time, cell_output, cell_state, loop_state):

      if cell_output is None:
        finished = initial_finished
        next_input = initial_decoder_input
        next_cell_state = initial_state
        emit_output = cell_output
        return (finished, next_input, next_cell_state, emit_output, 0)

      else:
        prediction = tf.argmax(tf.add(tf.matmul(cell_output,W), b), axis=1, output_type=tf.int32)
        finished_prediction_at_eos = tf.equal(prediction, eos_const)
        finished_over_time = tf.greater(time, max_time_const)
        finished = tf.logical_or(finished_prediction_at_eos, finished_over_time)
        prediction_embedded = tf.nn.embedding_lookup(W_embedding_layer_regular_words, prediction, name='decoder_input_embedding_lookup')
        next_input = prediction_embedded
        next_cell_state = cell_state
        emit_output = cell_output
        return (finished, next_input, next_cell_state, emit_output, loop_state + 1)

    raw_rnn_loop_fn = raw_rnn_loop_fn_train
    if mode == "extract":
      raw_rnn_loop_fn = raw_rnn_loop_fn_extract

    decoder_output_ta, _, _ = tf.nn.raw_rnn( decoder_stacked_rnn, raw_rnn_loop_fn, scope='Decoder')
    decoder_output = decoder_output_ta.stack()


info("DONE") # }}}

### Fully Connected Layer ### {{{
info("Set up fully connected layer...")

with tf.name_scope('fully_connected') as scope:
    decoder_output_reshape_for_ff_layer = tf.reshape(decoder_output, [-1, number_of_lstm_units_per_layer])
    ff_layer_neuron = tf.add(tf.matmul(decoder_output_reshape_for_ff_layer,W), b)
    ff_layer = tf.reshape(ff_layer_neuron, [-1, batch_size, vocabulary_size])
    extraction_word_indices = tf.argmax( ff_layer, axis=2, output_type=tf.int32)

info("DONE") # }}}

##############
### Train ### {{{
info("Set up loss and training function....")

#### Loss ####
with tf.name_scope('loss') as scope:
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.one_hot(y, depth=vocabulary_size, dtype=tf.float32), logits = ff_layer ) )
    loss_training = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                      weights=tf.transpose(W),
                      biases=b,
                      labels=tf.reshape(y, [-1, 1]),
                      inputs = decoder_output_reshape_for_ff_layer,
                      num_sampled = vocabulary_size/5,
                      num_classes = vocabulary_size,
                      num_true = 1 ))


#### Optimizer ####
with tf.name_scope('train') as scope:
    train_step = tf.train.AdamOptimizer().minimize(loss_training)


#### summarizers ####
    correct_prediction = tf.equal(y, tf.argmax( ff_layer, axis=2, output_type=tf.int32))
    not_empty_words = tf.not_equal(y, empty_const)
    # remove empty words from correct prediction
    correct_prediction = tf.equal(not_empty_words, correct_prediction)
    correct_prediction_count = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
    not_empty_words_count = tf.reduce_sum(tf.cast(not_empty_words, tf.float32))
    word_precision = tf.divide(correct_prediction_count, not_empty_words_count)
    word_precision_correct_words = correct_prediction_count
    word_precision_total_words = not_empty_words_count

    training_summary = tf.summary.merge( [tf.summary.scalar('word_precision', word_precision),
                                          tf.summary.scalar('loss_training', loss_training),
                                          ],name="training")

#### batch #### {{{
def generate_batch(X, Y, batch_size, start_index, node_size):

    include_target_data = True
    if(Y == None):
      include_target_data = False

    batch_encoder_input = []
    batch_decoder_output = []
    for i in range(batch_size):
      i_source_elem = (start_index + i) % len(X)
      batch_encoder_input.append( X[i_source_elem] )
      if include_target_data:
        batch_decoder_output.append( Y[i_source_elem]+ [word_to_index_regular_words['<eos>']])

    #convert to numpy
    def ctnp(ll):
      max_seq_len = max([len(i) for i in ll])
      ret = numpy.zeros([max_seq_len, batch_size])
      for i in range(batch_size):
        for j in range(max_seq_len):
          if len(ll[i]) > j:
            ret[j, i] = ll[i][j]
          else:
            break
      return ret

    #convert to numpy considering nodes
    def ctnpn(ll):
      max_seq_len = max([len(i) for i in ll])
      ret = numpy.zeros([max_seq_len, batch_size, node_size])
      for i in range(batch_size):
        for j in range(max_seq_len):
          if len(ll[i]) > j:
            ret[j, i,:] = ll[i][j]
          else:
            break
      return ret

    if include_target_data:
      return ctnpn(batch_encoder_input), ctnp(batch_decoder_output)
    else:
      return ctnpn(batch_encoder_input)

# }}}
info("DONE") # }}}


########################
### Training Session ### {{{
if mode == "train":
  info("TRAINING SESSION")

  def dropout_keep_f(i_epoch):
    d = [0.5, 0.6, 0.7 , 0.8, 0.8 , 0.8, 1 ]
    if i_epoch < len(d):
      return d[i_epoch]
    else:
      return d[-1]

  def feed_prediction_prob_f(i_epoch):
    d = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8 ]
    if i_epoch < len(d):
      return d[i_epoch]
    else:
      return d[-1]


  import shutil
  from sklearn.utils import shuffle

  with tf.Session() as session:

    info("session_name: " + session_name)
    i_epoch = 0

    saver = tf.train.Saver()
    if continue_session:
      info("load previous session...")
      saver.restore(session, "model/"+session_name+"/model")
      with open("model/"+session_name+"/i_epoch") as epochfile:
        i_epoch = int(epochfile.readline().strip()) +1
        print "continue at epoch " + str(i_epoch)
      info("DONE")
    else:
      info("initialize global variables...")
      session.run(tf.global_variables_initializer())
      try:
        shutil.rmtree('tensorboard/'+session_name)
      except:
        pass
      info("DONE")
    writer = tf.summary.FileWriter('tensorboard/'+session_name)
    writer.add_graph(session.graph)

    info("maximum number of epochs: "+ str(epochs))
    info("batches_per_epoch: " + str(batches_per_epoch))
    info("validation_stop: " + str(validation_stop))
    info("test_batches: " + str(test_batches))
    info("start training...")
    sys.stdout.write("INFO: epoch: 0 training iteration: 0")
    sys.stdout.flush()
    validation_loss_list = []
    validation_increase = 0

    while(i_epoch < epochs):

      ### shuffle the contents of the buckets ###
      for i in range(len(X_buckets)):
          X_buckets[i], Y_buckets[i] = shuffle(X_buckets[i],Y_buckets[i],random_state=333+i_epoch+i)

      ### shuffle order of buckets ###
      X_buckets, Y_buckets = shuffle(X_buckets, Y_buckets, random_state=345+i_epoch)

      X_train = []
      Y_train = []
      for i in range(len(X_buckets)):
          X_train += X_buckets[i]
          Y_train += Y_buckets[i]

      ### validate ###
      w_a_correct = 0
      w_a_total = 0
      loss_l = []
      validation_batches = len(X_validate)/batch_size
      for i in range(validation_batches):
        sys.stdout.write("\rINFO: epoch: " + str(i_epoch) + ", validation batch: " + str(i) + "............")
        sys.stdout.flush()
        batch = generate_batch(X_validate, Y_validate, batch_size, i*batch_size, number_of_words_per_DG_node)
        w_a_correct_, w_a_total_, l = session.run([word_precision_correct_words, word_precision_total_words, loss], feed_dict={x: batch[0], y: batch[1], decoder_input_ground_truth: batch[1], dropout_output_keep_prob: 1, feed_prediction_prob: [1]})
        loss_l.append(l)
        w_a_correct += w_a_correct_
        w_a_total += w_a_total_

      validation_word_precision = float(w_a_correct)/float(w_a_total)
      summary = tf.Summary(value=[
        tf.Summary.Value(tag="validation/word_precision", simple_value=validation_word_precision),
        ])
      writer.add_summary(summary, i_epoch)

      validation_loss = numpy.mean(loss_l)
      validation_loss_list.append(validation_loss)
      summary = tf.Summary(value=[
        tf.Summary.Value(tag="validation/loss", simple_value=validation_loss),
        ])
      writer.add_summary(summary, i_epoch)

      ### test ###
      w_a_correct = 0
      w_a_total = 0
      loss_l = []
      for i in range(test_batches):
        sys.stdout.write("\rINFO: epoch: " + str(i_epoch) + ", test batch: " + str(i) + "............")
        sys.stdout.flush()
        batch = generate_batch(X_test, Y_test, batch_size, i*batch_size, number_of_words_per_DG_node)
        w_a_correct_, w_a_total_,l = session.run([word_precision_correct_words, word_precision_total_words, loss], feed_dict={x: batch[0], y: batch[1], decoder_input_ground_truth: batch[1], dropout_output_keep_prob: 1, feed_prediction_prob: [1]})
        loss_l.append(l)
        w_a_correct += w_a_correct_
        w_a_total += w_a_total_

      test_word_precision=float(w_a_correct)/float(w_a_total)
      summary = tf.Summary(value=[
        tf.Summary.Value(tag="test/word_precision", simple_value=test_word_precision),
        ])
      writer.add_summary(summary, i_epoch)
      test_loss=numpy.mean(loss_l)
      summary = tf.Summary(value=[
        tf.Summary.Value(tag="test/loss", simple_value=test_loss),
        ])
      writer.add_summary(summary, i_epoch)

      sys.stdout.write("\rINFO: epoch: " + str(i_epoch) + "                                         " \
                          "\n\tvalidation loss: " + str(validation_loss) + \
                          "\n\tvalidation word precision: " + str(validation_word_precision) + \
                          "\n\ttest loss: " + str(test_loss) + \
                          "\n\ttest word precision:" + str(test_word_precision) + "\n")
      sys.stdout.flush()

      ### stop/save ###
      i_epoch += 1
      if validation_stop:
        if sorted(validation_loss_list)[0] < validation_loss_list[-1]:
          validation_increase += 1
          print "INFO: validation error increased " +str(validation_increase) + " times"
          if validation_increase >= validation_increase_tolerate:
            print "INFO: validation stop reached: increased " + str(validation_increase) + " times"
            break
        else:
          validation_increase = 0
          #save only model with smallest validation error
          if save_session:
            sys.stdout.write("\rINFO: Saving session..............")
            sys.stdout.flush()
            saver.save(session, "model/"+session_name+"/model")
            with open("model/"+session_name+"/i_epoch", "w") as epochfile:
              epochfile.write(str(i_epoch))
      else:
        #save
        if save_session:
          sys.stdout.write("\rINFO: Saving session..............")
          sys.stdout.flush()
          saver.save(session, "model/"+session_name+"/epoch_"+str(i_epoch)+"/model")
          with open("model/"+session_name+"/i_epoch", "w") as epochfile:
            epochfile.write(str(i_epoch))

      ### train ###
      number_of_batches_for_this_epoch = len(X_train)/batch_size
      if len(X_train)%batch_size != 0:
        number_of_batches_for_this_epoch +=1
      if batches_per_epoch > 0: # batches_per_epoch == 0 means all the batches
        number_of_batches_for_this_epoch = batches_per_epoch
      print "\nINFO: training batches: " + str(number_of_batches_for_this_epoch)

      for i in range(number_of_batches_for_this_epoch):

          iternumber = i_epoch*batches_per_epoch + i
          sys.stdout.write("\rINFO: epoch: " + str(i_epoch) + ", training batch: " + str(i) + "............")
          sys.stdout.flush()

          batch = generate_batch(X_train, Y_train, batch_size, i*batch_size, number_of_words_per_DG_node)

          summary, _ = session.run([training_summary, train_step], feed_dict={
              x: batch[0],
              y: batch[1],
              decoder_input_ground_truth: batch[1],
              dropout_output_keep_prob: dropout_keep_f(i_epoch),
              feed_prediction_prob: [feed_prediction_prob_f(i_epoch)]
              })
          writer.add_summary(summary, iternumber)

    print "\nINFO: DONE" # }}}


########################
### Extraction Session ### {{{
if mode == "extract":
  info("EXTRACTION SESSION")

  #### Load input sentences ####

  info("Read dependency graphs from file ...")

  with open(dependency_graphs_file) as dgf:
    conll06_ID = 0
    conll06_FORM = 1
    conll06_LEMMA = 2
    conll06_CPOSTAG = 3
    conll06_POSTAG = 4
    conll06_FEATS = 5
    conll06_HEAD_ID = 6
    conll06_DEPREL = 7
    conll06_PHEAD = 8
    conll06_PDEPREL = 9

    position_embedded_cols = [ conll06_ID, conll06_HEAD_ID ]
    regular_word_embedded_cols = [conll06_FORM, conll06_LEMMA]
    postag_deplabel_embedded_cols = [conll06_POSTAG, conll06_DEPREL]

    dep_graphs = []
    dep_graph = []
    placeholder_word_dicts = []
    placeholder_word_dict = {}

    def getWordIndex(word, placeholder_word_dict):
        # we have to use None instead of False here in order to not confuse index 0 with False
        i_word = word_to_index_regular_words.get(placeholder_word_dict.get(word, word), None)
        if i_word == None:
            placeholder_i = len(placeholder_word_dict)
            if placeholder_i < number_of_placeholder_words:
                placeholder_word_dict[word] = "<placeholder"+str(placeholder_i)+">"
            else:
                placeholder_word_dict[word] = "<unknown>"
        i_word = word_to_index_regular_words.get(placeholder_word_dict.get(word, word), None)
        assert i_word != None
        return i_word, placeholder_word_dict

    for line in dgf:
        row = line.split('\t')
        if len(line) == 1: # empty line between graphs
            dep_graphs.append(dep_graph)
            dep_graph = []
            placeholder_word_dicts.append(placeholder_word_dict)
            placeholder_word_dict = {}
        else:
            node = []
            for i in position_embedded_cols:
                if row[i] == '_':
                  node.append(0)
                else:
                  node.append(int(row[i]))
            for i in regular_word_embedded_cols:
                i_word, placeholder_word_dict = getWordIndex(row[i], placeholder_word_dict)
                node.append(i_word)
            for i in postag_deplabel_embedded_cols:
                node.append(word_to_index_postags_deplabels.get(row[i], word_to_index_postags_deplabels["_"]))
            dep_graph.append(node)
    if len(dep_graph) > 0:
      dep_graphs.append(dep_graph)
      placeholder_word_dicts.append(placeholder_word_dict)

  info("DONE")

  index_to_word = [ i[0] for i in sorted( word_to_index_regular_words.items(), key=lambda x: x[1]) ]

  def generate_input(X, batch_size, start_index, node_size):
    return generate_batch(X, None, batch_size, start_index, node_size)

  with tf.Session() as session:

    info("session_name: " + session_name)

    saver = tf.train.Saver()
    info("load model...")
    saver.restore(session, "model/"+session_name+"/model")
    info("DONE")

    for i in range(len(dep_graphs)):
      assert batch_size == 1
      batch = generate_input(dep_graphs, batch_size, i, number_of_words_per_DG_node)
      if len(batch) > max_number_of_nodes_per_DG:
        print "skip: sentence too long"
        continue
      output_indices = session.run(extraction_word_indices, feed_dict={x: batch })
      output_indices = [ int(w[0]) for w in output_indices]
      placeholders = {y:x for x,y in placeholder_word_dicts[i].iteritems() if y != '<unknown>'}
      extracted_words = [placeholders.get(index_to_word[i], index_to_word[i]) for i in output_indices]
      extraction_string = " ".join(extracted_words)

      print extraction_string

  info("DONE") # }}}

end_time = time.time()
timespan = end_time - start_time
hours = int(timespan/3600)
minutes = int((timespan % 3600)/60)
seconds = int(timespan %60)
info("total time: " +str(hours) + " hours, "+ str(minutes) + " minutes, "+ str(seconds) + " seconds")
