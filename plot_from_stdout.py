re_float = "[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

if __name__ == "__main__":
    import numpy as np
    import re

    import matplotlib
    matplotlib.use('agg')
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    # Command line parameters
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("stdout_files", type=str, nargs='+',
                        help="The files containing stdout from sampler.py")
    parser.add_argument("--plot", type=str, default="plot.pdf",
                        help="Path for writing the plot file (default: %(default)s)")
    args = parser.parse_args()

    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    # Read the data files and extract the reults
    data = []
    #exp = re.compile("step = (\d+) acceptance-rate = (%s) step-size = %s\s+log_prob = (%s) l2 = (%s)\s+train_loss = (%s) train_accuracy = (%s)\s+val_loss = (%s) val_accuracy = (%s)" % (re_float, re_float, re_float, re_float, re_float, re_float, re_float, re_float))

    exp = re.compile("step = (\d+) acceptance-rate = (%s) step-size = \[?(?:%s|,| )+\]?\s+log_prob = (%s) l2 = (%s)\s+train_loss = (%s) train_accuracy = (%s)\s+val_loss = (%s) val_accuracy = (%s)" % (re_float, re_float, re_float, re_float, re_float, re_float, re_float, re_float))

    for fn in args.stdout_files:
        print("Reading:", fn)
        with open(fn, "r") as fh:
            text = fh.read()

        parameters = {}
        for name, t in [("beta", float),
                        ("num_burnin_steps", int),
                        ("num_steps", int),
                        ("prior_scale", float),
                        ("train_size", int),
                        ("geo", int),
                        ("he", int),
                        ("dataset", str),
                        ("method", str)]:

            re_exp = "\S+" if t is str else re_float
            match = re.search("%s = (%s)" % (name, re_exp), text)
            parameters[name] = t(match.group(1)) if match else 0

        results = exp.findall(text)
        results = list(map(lambda l: list(map(float, l)), results))
        results = np.array(list(results))
        data.append((parameters, results.T))

    data.sort(key=lambda v: (v[0]['dataset'], v[0]['train_size'], v[0]['beta'], v[0]['prior_scale'], v[0]['geo']))
        
    # Plot the results
    with PdfPages(args.plot) as pdf:
        for parameters, results in data:
            fig, ax = plt.subplots(1, 6, figsize=(25, 6))
            plt.subplots_adjust(wspace=0.3)
            fig.suptitle(r"%s with %s at $\beta=%.2E$, $\sigma=%f$, n=%d, geo=%d, he=%d" % (parameters["method"].upper(), parameters["dataset"], parameters["beta"], parameters["prior_scale"], parameters["train_size"], parameters["geo"], parameters["he"]))

            ax[0].plot(results[0], results[1])
            ax[0].set_xlabel(r'epoch')
            ax[0].set_ylabel(r'Acceptance-rate')
            ax[0].set_ylim(0, 1)

            ax[1].plot(results[0], results[2])
            ax[1].set_xlabel(r'epoch')
            ax[1].set_ylabel(r'$\log P(\mathbf{w})$')

            ax[2].plot(results[0], np.sqrt(results[3]))
            ax[2].set_xlabel(r'epoch')
            ax[2].set_ylabel(r'$\frac{1}{2} ||\mathbf{w}||_2$')

            ax[3].plot(results[0], results[4], label="Training")
            ax[3].plot(results[0], results[6], label="Validation")
            ax[3].set_yscale('log')
            ax[3].set_xlabel(r'epoch')
            ax[3].set_ylabel(r'loss')
            ax[3].legend()

            ax[4].plot(results[0], results[5], label="Training")
            ax[4].plot(results[0], results[7], label="Validation")
            ax[4].set_xlabel(r'epoch')
            ax[4].set_ylabel(r'acc')
            ax[4].set_ylim(0, 1)
            ax[4].legend()

            ax[5].plot(results[0], results[5], label="Training")
            ax[5].plot(results[0], results[7], label="Validation")
            ax[5].set_xlabel(r'epoch')
            ax[5].set_ylabel(r'acc')
            ax[5].set_ylim(0.9, 1)
            ax[5].legend()
            
            pdf.savefig()
            plt.close()