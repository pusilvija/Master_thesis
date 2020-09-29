if __name__ == "__main__":
    import pickle
    import numpy as np
    import tensorflow as tf
    from sampler_CNN import CNN

    # Command line parameters
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pickle_files", type=str, nargs='+',
                        help="Filnames for reading the results of the test")
    parser.add_argument("--plot", type=str, default="plot.pdf",
                        help="Path for writing the plot file (default: %(default)s)")
    args = parser.parse_args()

    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    # Import
    import os.path
    from sklearn.calibration import calibration_curve
    import matplotlib
    matplotlib.use('agg')
    # from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    # Plotting with histograms
    fig, ax1 = plt.subplots(figsize=(10,10))
    for fn in args.pickle_files:
        name = os.path.splitext(os.path.basename(fn))[0]
        name = name.replace("_sample", "")
        name = name.replace("_fashionmnist", "_fmnist")
        name = name.replace("_ns5000000_nbs1000000_ss0.002", "")

        data = pickle.load(open(fn, 'rb'))

        p = np.array(data['acc_p']).max(1)
        y_true = np.array(tf.cast(tf.equal(data['y_val'], CNN.predict(data['acc_p'])), tf.int32))
        
        ax1.plot(*calibration_curve(y_true, p, n_bins=20,)[::-1], marker='o', label=name)
        
        ax2 = ax1.twinx()
        ax2.hist(p, bins=20,label=name, alpha=0.3)
        ax2.set_xlim(0,1)
        ax1.legend()


    ax1.plot([0,1],[0,1], c='black', ls='--')
    ax1.axis('equal')
    ax1.set_xlabel("Average predicted probability")
    ax2.set_ylabel("Count")
    ax1.set_ylabel("Relative frequency of positive examples")
    #ax.set_xscale('log')
    #ax.set_yscale('log')

    fig.savefig(args.plot)