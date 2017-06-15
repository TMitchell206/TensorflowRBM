"""
This script is a straight-forward implementation of a
Restricted-Boltzmann Machine (RBM), with a single gibbs
sampling step (to be parameterized). The script was written
becuase the author could not find a simple implementation
of a RBM, using raw tensorflow. It is up to the individual
to implement data handling, batch generation, calucating,
number of batches, etc...
"""

# HYPERPARAMETERS (global script)
num_visible = 5
num_hidden = 11
learning_rate = 0.01
h_unit_type = 'binary' # or 'guassian'

# PARAMETERS
learning_rate = 0.001
batch_size = 1
num_epochs = 50
stddev = 0.1

# PLACEHOLDERS
visible = tf.placeholder(tf.float32, [None, num_visible], name='X')
hrand = tf.placeholder(tf.float32, [None, num_hidden], name='hrand')
vrand = tf.placeholder(tf.float32, [None, num_visible], name='vrand')

x = tf.placeholder(tf.float32, [None, num_visible])
#y = tf.placeholder(tf.float32, [None, NUM_CLASSES])

# VARIABLES
W  = tf.Variable(tf.random_normal((num_visible, num_hidden), mean=0.0, stddev=stddev), name='weights')
bh = tf.Variable(tf.zeros([num_hidden]), name='hidden_bias')
bv = tf.Variable(tf.zeros([num_visible]), name='visible_bias')

def rbm_model(x, W, bv, bh, h_unit_type='binary'):

    pos_hidden_activations = tf.nn.bias_add(tf.matmul(x, W), bh)
    pos_hidden_probs = tf.sigmoid(pos_hidden_activations)
    pos_hidden_states = tf.nn.relu(tf.sign(pos_hidden_probs - np.random.rand(1, num_hidden)))#tf.to_float(tf.greater_equal(0.5,  pos_hidden_probs, name=None))
    pos_associations = tf.matmul(tf.transpose(x), pos_hidden_probs)

    neg_visible_activations = tf.nn.bias_add(tf.matmul(pos_hidden_states, tf.transpose(W)), bv)
    if h_unit_type is 'gauss':
         neg_visible_probs = tf.truncated_normal((1, num_visible), mean=neg_visible_activations, stddev=stddev)
    else:
        neg_visible_probs = tf.sigmoid(neg_visible_activations)
    neg_hidden_activations = tf.nn.bias_add(tf.matmul(neg_visible_probs, W), bh)
    neg_hidden_probs = tf.sigmoid(neg_hidden_activations)
    neg_hidden_states = tf.nn.relu(tf.sign(pos_hidden_probs - np.random.rand(1, num_hidden)))
    neg_associations = tf.matmul(tf.transpose(neg_visible_probs), neg_hidden_probs)

     out = {'pos_assoc': pos_associations,
            'neg_assoc': neg_associations,
            'hidden_pobs': pos_hidden_probs,
            'reconstruction': neg_visible_probs,
            'pos_hidden_probs': pos_hidden_probs,
            'pos_hidden_states': pos_hidden_states,
            'neg_visible_probs': neg_visible_probs,
            'neg_hidden_probs': neg_hidden_probs,
            'neg_hidden_states': neg_associations,
            'weights': W,
            'bh': bh,
            'bv': bv }

     return out

rbm_results = rbm_model(x, W, bv, bh)
reconstruction = rbm_results['reconstruction']
positive = rbm_results['pos_assoc']
negative = rbm_results['neg_assoc']
hprobs0 = rbm_results['pos_hidden_probs']
hstates0 = rbm_results['pos_hidden_states']
vprobs = rbm_results['neg_visible_probs']
hprobs1 = rbm_results['neg_hidden_states']
hstates1 = rbm_results['neg_hidden_states']

loss = tf.reduce_sum(tf.pow(x - vprobs, 2))

W_update = W.assign_add( learning_rate*(positive-negative) )
bh_update = bh.assign_add( learning_rate*tf.reduce_mean(hprobs0-hprobs1, 0) )
bv_update = bv.assign_add( learning_rate*tf.reduce_mean(x-vprobs, 0) )
updates = [W_update, bh_update, bv_update]

with tf.Session() as sess:
    #init variables:
    sess.run(tf.global_variables_initializer())
    print sess.run(rbm_results['weights'])
    print sess.run(rbm_results['bh'])
    print sess.run(rbm_results['bv'])

    print 'TRAINING STARTED'
    for i in range(num_epochs):
        ave_loss = 0
        np.random.shuffle(x_train)
        for k in range(len(num_batches)):
            sess.run(updates, feed_dict={x: batch)})
            ave_loss = ave_loss + sess.run(loss, feed_dict={x: np.array([x_train[k]])})/(len(x_train))
        print ave_loss

    print 'TRAINING COMPLETED'
