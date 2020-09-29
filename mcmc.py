import tensorflow as tf
import tensorflow_probability as tfp


# A MCMC transision kernel for tracking the simulation
class StepTracer(tfp.python.mcmc.kernel.TransitionKernel):
    def __init__(self, inner_kernel, trace_fn, trace_freq=1, name=None):
        inner_kernel = tfp.python.mcmc.internal.util.enable_store_parameters_in_results(inner_kernel)
        self._parameters = dict(
            inner_kernel=inner_kernel,
            trace_fn=trace_fn,
            trace_freq=trace_freq,
            name=name
        )

    @property
    def inner_kernel(self):
        return self._parameters['inner_kernel']

    @property
    def name(self):
        return self._parameters['name']

    @property
    def trace_fn(self):
        return self._parameters['trace_fn']

    @property
    def trace_freq(self):
        return self._parameters['trace_freq']

    @property
    def parameters(self):
        """Return `dict` of ``__init__`` arguments and their values."""
        return self._parameters

    @tf.function
    def one_step(self, current_state, previous_kernel_results):
        # Step the inner kernel.
        new_state, new_kernel_results = self.inner_kernel.one_step(
            current_state, previous_kernel_results)
        if new_kernel_results.step % self.trace_freq == 0:
            self.trace_fn(new_state, new_kernel_results)
        return new_state, new_kernel_results

    def is_calibrated(self):
        return self.inner_kernel.is_calibrated()

    def bootstrap_results(self, init_state):
        inner_results = self.inner_kernel.bootstrap_results(init_state)
        return inner_results