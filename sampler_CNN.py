import tensorflow as tf
import sys

# Define the model
class CNN:
    def __init__(self, filter_shapes, dense_nodes, input_shape, act_fn=tf.nn.relu):
        self.filter_shapes = filter_shapes
        self.dense_nodes = dense_nodes
        self.act_fn = act_fn

        # Calculate the output shape of the CNN
        tmp_vars = self.initialize_vars(only_cnn=True)
        x = self.call(tf.zeros(shape=[1]+input_shape), *tmp_vars, only_cnn=True)
        self.cnn_output_shape = x.shape[-1]

    @tf.function
    def call(self, x, *vars, only_cnn=False):
        # Check that the number of parameters are correct
        if only_cnn:
            assert(len(vars) == 2 * len(self.filter_shapes))
        else:
            assert(len(vars) == 2 * len(self.filter_shapes) + 2 * len(self.dense_nodes))

        # CNN layers
        for i in range(0, 2*len(self.filter_shapes), 2):
            strides = (1, 1) if i == 0 else (2, 2)
            x = self.act_fn(tf.nn.bias_add(tf.nn.conv2d(input=x, filters=vars[i], strides=strides, padding="VALID"), vars[i+1]))

        x = tf.reshape(x, [x.shape[0], -1])

        if only_cnn:
            return x

        # Dense layers
        for i in range(2*len(self.filter_shapes), len(vars), 2):
            x = tf.matmul(x, vars[i]) + vars[i+1]
            if i < len(vars)-2:
                x = self.act_fn(x)

        return x

    def initialize_vars(self, initializer=tf.initializers.he_normal(), only_cnn=False):
        vars = []
        for i, filter_shape in enumerate(self.filter_shapes):
            w = tf.Variable(initializer(shape=filter_shape), name="W%d" % i)
            b = tf.Variable(tf.zeros(shape=filter_shape[-1]), name="b%d" % i)
            vars.extend([w, b])

        if only_cnn:
            return vars

        previous_nodes = self.cnn_output_shape
        for i, nodes in enumerate(self.dense_nodes):
            n = i+len(self.filter_shapes)
            w = tf.Variable(initializer(shape=(previous_nodes, nodes)), name="W%d" % n)
            b = tf.Variable(tf.zeros(shape=(nodes)), name="b%d" % n)

            vars.extend([w, b])
            previous_nodes = nodes

        return vars

    @staticmethod
    @tf.function
    def loss(y, logits):
        return tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(y, logits))

    @staticmethod
    @tf.function
    def weights(*vars):
        return vars[::2]

    @staticmethod
    @tf.function
    def l2(*vars):
        return tf.add_n([tf.nn.l2_loss(w) for w in CNN.weights(*vars)])

    @staticmethod
    @tf.function
    def predict(logits):
        return tf.argmax(logits, axis=1, output_type=tf.int32)

    @staticmethod
    @tf.function
    def accuracy(y, logits):
        return tf.reduce_mean(tf.cast(tf.equal(y, CNN.predict(logits)), tf.float32))

# Define target distribution
def get_unnormalized_log_prob(model, x, y, beta, prior_scale, geo, he, prior_scales_with_beta, likelihood_scales_with_beta):
    def unnormalized_log_prob(*vars):
        logits = model.call(x, *vars)
        loss = model.loss(y, logits)
        l2 = model.l2(*vars)
        log_likelihood = -loss

        prior = 0
        if he:
            for w in model.weights(*vars):
                var = 2. / np.prod(w.shape[:-1])  # He initialization
                prior += -tf.nn.l2_loss(w)/var # l2_loss includes 1/2
        elif prior_scale > 0:
            prior += -l2/prior_scale**2 # l2_loss includes 1/2
        if geo:
            d = sum(map(lambda w: np.prod(w.shape), model.weights(*vars)))
            prior += -(d - 1) * tf.math.log(2 * l2) / 2
            
        if prior_scales_with_beta:
            prior = beta*prior
        if likelihood_scales_with_beta:
            log_likelihood = beta*log_likelihood

        return log_likelihood + prior
    return unnormalized_log_prob

def get_logits(model,x, *vars):
    return model.call(x, *vars)

def entropy(logits):
    logp = logits - logsumexp(logits, axis=1, keepdims=True)
    return  -np.sum( np.exp(logp)*logp, axis=1) 

def p(logits):
    logp = logits - logsumexp(logits, axis=1, keepdims=True)
    return  np.exp(logp) 

# A trace function that calculating training and validation loss
class Tracer:
    def __init__(self, model, unnormalized_log_prob, x_train, y_train, x_val, y_val):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.n_accept = tf.Variable(tf.zeros(shape=(), dtype=tf.int64))
        self.n_attempt = tf.Variable(tf.zeros(shape=(), dtype=tf.int64))

    def call(self, state, kernel_results):
        log_prob = unnormalized_log_prob(*state)
        l2 = model.l2(*state)

        train_logits = model.call(self.x_train, *state)
        train_loss = model.loss(self.y_train, train_logits)
        train_accuracy = model.accuracy(self.y_train, train_logits)

        val_logits = model.call(self.x_val, *state)
        val_loss = model.loss(self.y_val, val_logits)
        val_accuracy = model.accuracy(self.y_val, val_logits)

        self.n_accept.assign_add(tf.cast(kernel_results.inner_results.is_accepted, tf.int64))
        self.n_attempt.assign_add(1)
        acceptance_rate = tf.cast(self.n_accept, tf.float32) / tf.cast(self.n_attempt, tf.float32)

        tf.print("step =", kernel_results.step, "acceptance-rate =", acceptance_rate, "step-size =", kernel_results.new_step_size, output_stream=sys.stdout)
        tf.print("\tlog_prob =", log_prob, "l2 =", l2, output_stream=sys.stdout)
        tf.print("\ttrain_loss =", train_loss, "train_accuracy =", train_accuracy, output_stream=sys.stdout)
        tf.print("\tval_loss =", val_loss, "val_accuracy =", val_accuracy, output_stream=sys.stdout)

        if hasattr(kernel_results.inner_results, 'leapfrogs_taken'):
            tf.print("\tleapfrogs_taken =", kernel_results.inner_results.leapfrogs_taken, output_stream=sys.stdout)
        

if __name__ == "__main__":
    import pickle
    import numpy as np
    import tensorflow as tf
    import tensorflow_probability as tfp
    from tensorflow import keras
    from mcmc import StepTracer
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.special import logsumexp
    import ternary
    from matplotlib import cm
    from numpy import linspace


    # Command line parameters
    import argparse
    import distutils.util

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-steps", type=int, default=100000,
                        help="Number of steps (epochs) to run the HMC sampler (default: %(default)s)")
    parser.add_argument("--num-burnin-steps", type=int, default=1000,
                        help="Number of burnin steps (epochs) for the HMC sampler (default: %(default)s)")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Beta-value for the target distribution (default: %(default)s)")
    parser.add_argument("--prior-scale", type=float, default=0.1,
                        help="The scale for the Normal prior on weights (default: %(default)s)")
    parser.add_argument("--trace-freq", type=int, default=10,
                        help="Frequency for reporting training and validation loss (default: %(default)s)")
    parser.add_argument("--results-freq", type=int, default=100,
                        help="Frequency for reporting results (samples) from the simulation (default: %(default)s)")
    parser.add_argument("--train-size", type=int, default=None,
                        help="Only read this number of images for the training set (default: %(default)s)")
    parser.add_argument("--train-start-index", type=int, default=0,
                        help="Where in the training set to start (default: %(default)s)")
    parser.add_argument("--method", choices=['hmc', 'nuts', 'adam'], default="hmc",
                        help="Which method to use")
    parser.add_argument("--step-size", type=float, default=0.05,
                        help="Step size for HMC/ST (default: %(default)s)")
    parser.add_argument("--pickle", type=str, default=None,
                        help="Filname for dumping data and for reading when testing pickle (default: %(default)s)")
    parser.add_argument("--test-out-pickle", type=str, default=None,
                        help="Filname for dumping the results of the test (default: %(default)s)")
    parser.add_argument("--mode", choices=['sample', 'test', 'init'], default="sample",
                        help="Which mode to use")
    parser.add_argument("--num-leapfrog-steps", type=int, default=3,
                        help="Number of leapfrog steps for the HMC sampler (default: %(default)s)")
    parser.add_argument("--geo", type=distutils.util.strtobool, default='False',
                        help="Use the geometric term (default: %(default)s)")
    parser.add_argument("--he", type=distutils.util.strtobool, default='False',
                        help="Use the He Gaussian prior (default: %(default)s)")
    parser.add_argument("--prior-scales-with-beta", type=distutils.util.strtobool, default='False',
                        help="If the log-prior should be multiplied by beta (default: %(default)s)")
    parser.add_argument("--likelihood-scales-with-beta", type=distutils.util.strtobool, default='False',
                        help="If the log-likelihood should be multiplied by beta (default: %(default)s)")
    parser.add_argument("--test-stride", type=int, default=1,
                        help="Stride used in test mode (default: %(default)s)")
    parser.add_argument("--dataset", choices=['mnist', 'fashionmnist', 'cifar10', 'cifar10-gs', 'cifar100', 'cifar100-gs'], default='mnist',
                        help="which dataset to use (default: %(default)s)")
    parser.add_argument("--size-dense-layer", type=int, default=10,
                        help="Size of dense layer prior to output (default: %(default)s)")
    parser.add_argument("--act-fn", choices=['relu', 'selu', 'tanh', 'sigmoid'], default="relu",
                        help="Which activation function to use")
    parser.add_argument("--max-tree-depth", type=int, default=10,
                        help="Maximal tree depth for the NUTS sampler (default: %(default)s)")
    parser.add_argument("--name", type=str, default='no_name',
                        help="Name of the experiment (default: %(default)s)")
    parser.add_argument("--num-of-inits", type=int, default=1,
                        help="Number of inits (default: %(default)s)")
    parser.add_argument("--save_entropies", type=distutils.util.strtobool, default='False',
                        help="If need to save entropies of prior predictive (default: %(default)s)")
    parser.add_argument("--plot_simplex", type=distutils.util.strtobool, default='False',
                    help="If need to plot probability simplex for the first three classes (default: %(default)s)")
                            


    args = parser.parse_args()

    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    # Read MNIST dataset and scale the values to [0,1]
    if args.dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        output_size = 10
    elif args.dataset == "fashionmnist":
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        output_size = 10
    elif args.dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()
        output_size = 10
    elif args.dataset == "cifar10-gs":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

        # Remove extra dimension in labels
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()

        # Convert to gray scale
        x_train = tf.image.rgb_to_grayscale(x_train).numpy()
        x_test = tf.image.rgb_to_grayscale(x_test).numpy()
        output_size = 10
    elif args.dataset == "cifar100":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode='fine')
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()
        output_size = 100
    elif args.dataset == "cifar100-gs":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode='fine')

        # Remove extra dimension in labels
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()

        # Convert to gray scale
        x_train = tf.image.rgb_to_grayscale(x_train).numpy()
        x_test = tf.image.rgb_to_grayscale(x_test).numpy()

        output_size = 100

    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    max_train_size = x_train.shape[0]//6*5
    x_val = x_train[max_train_size:]
    y_val = y_train[max_train_size:]
    x_train = x_train[:max_train_size]
    y_train = y_train[:max_train_size]

    # Limit the size of the training set
    if args.train_size is not None:
        x_train, y_train = x_train[args.train_start_index:args.train_size], \
                           y_train[args.train_start_index:args.train_size]

    # Standartisation per channel
    x_data = [x_train, x_val, x_test]
    for dataset in x_data:
      # loop through channels
      for i in range(dataset.shape[-1]):
          mean = tf.math.reduce_mean(dataset[:,:,:,i], axis = [0,1,2])
          std = tf.math.reduce_std(dataset[:,:,:,i], axis = [0,1,2])
          dataset[:,:,:,i] = (dataset[:,:,:,i] - mean)/std

    print("Training set size:", x_train.shape[0])
    print("Validation set size:", x_val.shape[0])

    # Define a simple MNIST CNN model and set the target probability
    act_fn = {'relu': tf.nn.relu, 'selu': tf.nn.selu, 'tanh': tf.nn.tanh, 'sigmoid': tf.nn.sigmoid}[args.act_fn]
    model = CNN(filter_shapes=[(3, 3, x_train.shape[-1], 4), (3, 3, 4, 4), (3, 3, 4, 4)], dense_nodes=[args.size_dense_layer, output_size], input_shape=list(x_train.shape[1:4]), act_fn=act_fn)
    unnormalized_log_prob = get_unnormalized_log_prob(model, x_train, y_train, beta=args.beta, prior_scale=args.prior_scale, geo=args.geo, he=args.he, prior_scales_with_beta=args.prior_scales_with_beta, likelihood_scales_with_beta=args.likelihood_scales_with_beta)

    if args.mode == 'init':
      init_entropies = []
      init_probs = []
      for i in range(args.num_of_inits):
          if args.he or args.geo:
              initializer = tf.initializers.he_normal()
          else:
              initializer = tf.random_normal_initializer(stddev=args.prior_scale)
          vars = model.initialize_vars(initializer=initializer)

          # Get logits
          lgts = get_logits(model, x_train, *vars)

          # Get first three classes output probabilities
          pr = p(logits=lgts[:,:3])
          init_probs.append(pr)

          # Get entropy
          H_new = entropy(lgts)
          init_entropies.extend(H_new)

          print("###", i*100/args.num_of_inits,"%")

      if args.save_entropies == True:
        with open('entropies_' + args.name + '.txt', 'w') as f:
            for item in init_entropies:
              f.write("%s\n" % item)

      if args.plot_simplex == True:
        ### Scatter Plot SIMPLEX
        scale = 1.0
        cm_subsection = linspace(0, 1, args.num_of_inits) 
        colors = [ cm.viridis(x) for x in cm_subsection ]
        figure, tax = ternary.figure(scale=scale)
        figure.set_size_inches(10, 9)
        # tax.set_title(r"$\alpha$" + '=' + str(args.alpha), fontsize=35, y=1.08)  
        tax.boundary(linewidth=1.5)
        tax.gridlines(multiple=0.1,color="grey",linewidth=0.8)
        for i in range(len(init_probs)):
          # Plot a few different styles with a legend
          points = np.asarray(init_probs[i])
          tax.scatter(points, marker='o', color=colors[i], alpha=0.4, s=0.6)
        tax.ticks(axis='lbr', linewidth=1, fontsize=19, multiple=0.1, ticks=['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'],offset=0.02)
        tax.get_axes().axis('off')
        tax._redraw_labels()

        ternary.plt.savefig('simplex_' + args.name + '.png')

    else:
      # Initalize the variables
      if args.he or args.geo:
          initializer = tf.initializers.he_normal()
      else:
          initializer = tf.random_normal_initializer(stddev=args.prior_scale)
      vars = model.initialize_vars(initializer=initializer)

    n_weights = sum(map(lambda w: np.prod(w.shape), model.weights(*vars)))
    print("Number of weights:", n_weights)
    
    # Construct a tracer for the StepTracer
    tracer = Tracer(model, unnormalized_log_prob, x_train, y_train, x_val, y_val)

    if args.mode == "sample":
        if args.method == "hmc":
            # Initialize the HMC transition kernel.
            adaptive_hmc = StepTracer(
                tfp.mcmc.SimpleStepSizeAdaptation(
                    tfp.mcmc.HamiltonianMonteCarlo(
                        target_log_prob_fn=unnormalized_log_prob,
                        num_leapfrog_steps=args.num_leapfrog_steps,
                        step_size=args.step_size),
                    num_adaptation_steps=args.num_burnin_steps),  # We could have used 0.8*args.num_burnin_steps
                trace_fn=tracer.call,
                trace_freq=args.trace_freq
            )

            # Define function for running the chain (with decoration)
            @tf.function
            def run_chain():
                # Run the chain (with burn-in).
                samples, trace = tfp.mcmc.sample_chain(
                    num_results=args.num_steps // args.results_freq + 1,
                    num_burnin_steps=args.num_burnin_steps,
                    num_steps_between_results=args.results_freq-1,
                    current_state=vars,
                    kernel=adaptive_hmc)
                return samples, trace

            # Run the chain
            samples, trace = run_chain()

            print("Number of samples:", samples[0].shape[0])

            if args.pickle is not None:
                pickle.dump((samples, trace), open(args.pickle, "wb"))

        if args.method == "nuts":

            # Initialize the NUTS transition kernel.
            adaptive_nuts = StepTracer(
                tfp.mcmc.DualAveragingStepSizeAdaptation(
                    tfp.mcmc.NoUTurnSampler(
                        target_log_prob_fn=unnormalized_log_prob,
                        step_size=args.step_size,
                        max_tree_depth=args.max_tree_depth),
                    num_adaptation_steps=args.num_burnin_steps, # We could have used 0.8*args.num_burnin_steps
                    step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(step_size=new_step_size),
                    step_size_getter_fn=lambda pkr: pkr.step_size,
                    log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio,
                ), 
                trace_fn=tracer.call,
                trace_freq=args.trace_freq
            )

            # Define function for running the chain (with decoration)
            @tf.function
            def run_chain():
                # Run the chain (with burn-in).
                samples, trace = tfp.mcmc.sample_chain(
                    num_results=args.num_steps // args.results_freq + 1,
                    num_burnin_steps=args.num_burnin_steps,
                    num_steps_between_results=args.results_freq-1,
                    current_state=vars,
                    kernel=adaptive_nuts)
                return samples, trace

            # Run the chain
            samples, trace = run_chain()

            print("Number of samples:", samples[0].shape[0])

            if args.pickle is not None:
                pickle.dump((samples, trace), open(args.pickle, "wb"))

        elif args.method == "adam":

            optimizer = tf.keras.optimizers.Adam(learning_rate=args.step_size)

            for i in range(args.num_steps):

                with tf.GradientTape() as tape:

                    # y = model.call(x_train, *weights)

                    loss = -unnormalized_log_prob(*vars)

                gradients = tape.gradient(loss, vars)
                optimizer.apply_gradients(zip(gradients, vars))

                if i % args.trace_freq == 0:

                    # This evaluation on the training set could be optimized away
                    train_logits = model.call(x_train, *vars)
                    train_loss = model.loss(y_train, train_logits)
                    train_accuracy = model.accuracy(y_train, train_logits)
                    l2 = model.l2(*vars)

                    val_logits = model.call(x_val, *vars)
                    val_loss = model.loss(y_val, val_logits)
                    val_accuracy = model.accuracy(y_val, val_logits)

                    print("step =", i, "acceptance-rate =", 0, "step-size =", 0, "\tlog_prob = %.5f" % float(loss), "l2 = %.5f" % float(l2), "\ttrain_loss = %.5f" % float(train_loss), "train_accuracy = %.5f" % float(train_accuracy), "\tval_loss = %.5f" % float(val_loss) , "val_accuracy = %.5f" % float(val_accuracy))


            if args.pickle is not None:
                pickle.dump(([tf.expand_dims(v, 0) for v in vars], []), open(args.pickle, "wb"))

        else:
            raise "Unknown method: " + args.method
    elif args.mode == "test":
        samples, trace = pickle.load(open(args.pickle, "rb"))

        counter = 0
        idxs = range(0, samples[0].shape[0], args.test_stride)
        len_idxs = len(idxs)
        acc_p = None

        for i in idxs:
            vars = [v[i] for v in samples]

            p = tf.nn.softmax(model.call(x_val, *vars))

            acc_p = p if acc_p is None else acc_p + p
            counter += 1

            if counter % 10 == 0:
                print("i =", counter, "/", len_idxs, "[%.1f%%]" % (100*counter/len_idxs))

        acc_p = acc_p / len_idxs

        if args.test_out_pickle is not None:
            pickle.dump({'acc_p': acc_p,
                         'y_val': y_val,
                         'pickle': args.pickle,
                         'idxs': list(idxs)},
                        open(args.test_out_pickle, "wb"))
    elif args.mode == "init":
        pass
    else:
        raise "Unknown mode: " + args.mode
