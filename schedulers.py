import tensorflow as tf
from enum import Enum

# NOT FOR TRAINING. Only for learning rate search
class LearningRateSearchingSchedulePow(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, init_learning_rate = 1.0e-6, end_learning_rate = 1.0e-1, steps = 100) -> None:
        super().__init__()
        self._exp_base = tf.math.pow(end_learning_rate / init_learning_rate, 1.0/steps)
        self._init_lr = init_learning_rate

    def __call__(self, step):
        lr = tf.math.pow(self._exp_base, tf.cast(step, tf.float32)) * self._init_lr
        tf.summary.scalar('learning_rate', data=lr, step=tf.cast(step, tf.int64))
        return lr

# NOT FOR TRAINING. Only for learning rate search
class LearningRateSearchingScheduleLinear(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, init_learning_rate = 1.0e-6, end_learning_rate = 1.0e-1, steps = 100) -> None:
        super().__init__()
        self._delta = tf.math.divide(end_learning_rate - init_learning_rate, steps)
        self._init_lr = init_learning_rate

    def __call__(self, step):
        lr = self._delta * step + self._init_lr
        tf.summary.scalar('learning_rate', data=lr, step=tf.cast(step, tf.int64))
        return lr


class OneCycleLR(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, tot_iter, lr_range=(1.5e-5, 5.0e-2), step_size=5, tail=0.1, annealing_lr_fraction=0.1) -> None:
        super().__init__()
        self._base_lr = lr_range[0]
        self._max_lr = lr_range[1]
        self._amplitude_pk2pk = self._max_lr - self._base_lr
        self._step_size = step_size
        self._annealing_lr_frac = annealing_lr_fraction

        self._tot_steps = tf.math.floor(tf.cast(tf.math.divide(tot_iter, step_size), tf.float32))
        self._half_period = tf.math.floor(self._tot_steps * (1 - tail) / 2.0)
        self._delta = self._amplitude_pk2pk / self._half_period
        self._annealing_iters = self._tot_steps - 2 * self._half_period

    def __call__(self, step):
        wave_step = tf.cast(tf.floor(step/ self._step_size), tf.float32)
        # Condition to distinguish cycle and annealing sections
        condition = tf.less(wave_step, tf.cast(2.0*self._half_period, tf.float32))

        lr = tf.cond(pred=condition,
                    true_fn = lambda: - self._delta * tf.math.abs(tf.math.mod(wave_step, 2*self._half_period) - self._half_period) + self._amplitude_pk2pk + self._base_lr,
                    false_fn = lambda: self._base_lr - ((1.0-self._annealing_lr_frac)*self._base_lr)/self._annealing_iters * (wave_step - 2 * self._half_period)
                    )

        tf.summary.scalar('learning_rate', data=lr, step=tf.cast(step, tf.int64))
        return lr


class CyclicLR(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, tot_iter, lr_range=(1.5e-5, 5.0e-2), step_size=5, num_cycles=4, decay_func=lambda x, it: x) -> None:
        super().__init__()
        self._base_lr = lr_range[0]
        self._max_lr = lr_range[1]
        self._lr = self._base_lr
        self._step_size = step_size
        self._decay_func = decay_func

        self._tot_steps = tf.math.floor(tf.cast(tf.math.divide(tot_iter, step_size), tf.float32))
        self._period_steps = self._tot_steps // num_cycles
        self._half_period = tf.cast(tf.math.floor(self._period_steps / 2.0), tf.float32)

    def __call__(self, step):
        wave_step = tf.cast(tf.math.floor(step/ self._step_size), tf.float32)

        cycle = wave_step // tf.cast(2.0 * self._half_period, tf.float32)
        amplitude_pk = self._decay_func(self._max_lr - self._base_lr, cycle)
        delta = amplitude_pk / self._half_period

        lr = - delta * tf.math.abs(tf.math.mod(wave_step, 2*self._half_period) - self._half_period) + amplitude_pk + self._base_lr
        tf.summary.scalar('learning_rate', data=lr, step=tf.cast(step, tf.int64))
        return lr

if __name__=="__main__":
    len_cycle = 1508
    step_size = 20
    num_cycles = 4
    lr_range = (1.5e-5, 5.0e-2)
    # a = OneCycleLR(len_cycle, lr_range, step_size=step_size, tail=0.1)
    a = CyclicLR(tot_iter=len_cycle, lr_range=lr_range, step_size=step_size,
                num_cycles=num_cycles,
                decay_func=lambda x, it: x * 2**(-it))

    # Testing
    elems = []
    for i in range(len_cycle):
        elems.append(a(i))

    import matplotlib.pyplot as plt
    plt.plot(elems)
    plt.show()
    plt.savefig("plot.png")