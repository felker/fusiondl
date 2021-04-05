import src.global_vars as g
from src.utils.processing import concatenate_sublists
from src.utils.evaluation import get_loss_from_list

from src.core.trainercore import trainercore
from src.models.tensorflow import TCN, FRNN, LossCalculator, AccuracyCalculator

# from src.models.loader import ProcessGenerator
# from src.utils.state_reset import reset_states


import os
import sys
import time
import datetime
import numpy as np

from tensorflow.python.client import timeline
from functools import partial
from copy import deepcopy

# TODO(KGF): remove the next 3 lines?
# import keras sequentially because it otherwise reads from ~/.keras/keras.json
# with too many threads:
# from mpi_launch_tensorflow import get_mpi_task_index

# set global variables for entire module regarding MPI & GPU environment
g.init_GPU_backend(conf)
# moved this fn/init call to client-facing mpi_learn.py
# g.init_MPI()
# TODO(KGF): set "mpi_initialized" global bool flag?

g.flush_all_inorder()   # see above about conf_parser.py stdout writes

# initialization code for mpi_runner.py module:
if g.NUM_GPUS > 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(g.MY_GPU)
    # ,mode=NanGuardMode'

# "tensorflow" package on PyPI has GPU support as of 2.1.0
g.tf_ver = parse_version(get_distribution('tensorflow').version)

# g.flush_all_inorder()
# g.pprint_unique(conf)
# g.flush_all_inorder()
# g.comm.Barrier()



class Averager(object):
    """Compute and store a cumulative moving average (CMA).

    """

    def __init__(self):
        self.steps = 0
        self.cma = 0.0

    def add_val(self, new_val):
        self.cma = (self.steps * self.cma + 1.0 * new_val)/(self.steps + 1.0)
        self.steps += 1

    def get_ave(self):
        return self.cma


class tf_trainer(trainercore):
    def __init__(self, model, optimizer, batch_iterator, batch_size,
                 num_replicas=None, warmup_steps=1000, lr=0.01,
                 num_batches_minimum=100, conf=None):
        random.seed(g.task_index)
        np.random.seed(g.task_index)
        self.conf = conf
        self.start_time = time.time()
        self.epoch = 0
        self.num_so_far = 0
        self.num_so_far_accum = 0
        self.num_so_far_indiv = 0
        self.model = model
        self.optimizer = optimizer
        self.max_lr = 0.1

        self.batch_size = batch_size
        self.batch_iterator = batch_iterator
        self.set_batch_iterator_func()
        self.warmup_steps = warmup_steps
        self.num_batches_minimum = num_batches_minimum
        self.history = tf.keras.callbacks.History()
        self.model.stop_training = False
        if (num_replicas is None or num_replicas < 1
                or num_replicas > self.num_workers):
            self.num_replicas = self.num_workers
        else:
            self.num_replicas = num_replicas
        self.lr = (lr/(1.0 + self.num_replicas/100.0) if (lr < self.max_lr)
                   else self.max_lr/(1.0 + self.num_replicas/100.0))

    def set_batch_iterator_func(self):
        if (self.conf is not None
                and 'use_process_generator' in conf['training']
                and conf['training']['use_process_generator']):
            self.batch_iterator_func = ProcessGenerator(self.batch_iterator())
        else:
            self.batch_iterator_func = self.batch_iterator()

    def close(self):
        # TODO(KGF): extend __exit__() fn capability when this member
        # = self.batch_iterator() (i.e. is not a ProcessGenerator())
        if (self.conf is not None
                and 'use_process_generator' in conf['training']
                and conf['training']['use_process_generator']):
            self.batch_iterator_func.__exit__()

    def set_lr(self, lr):
        self.lr = lr

    def compile(self, optimizer, clipnorm, loss='mse'):
        # TODO(KGF): why was self.DUMMY_LR used in this function?
        if optimizer == 'sgd':
            optimizer_class = tf.keras.optimizers.SGD(lr=self.lr, clipnorm=clipnorm)
        elif optimizer == 'momentum_sgd':
            optimizer_class = tf.keras.optimizers.SGD(lr=self.lr, clipnorm=clipnorm,
                                  decay=1e-6, momentum=0.9)
        elif optimizer == 'adam':
            optimizer_class = tf.keras.optimizers.Adam(lr=self.lr, clipnorm=clipnorm)
        elif optimizer == 'rmsprop':
            optimizer_class = tf.keras.optimizers.RMSprop(lr=self.lr, clipnorm=clipnorm)
        elif optimizer == 'nadam':
            optimizer_class = tf.keras.optimizers.Nadam(lr=self.lr, clipnorm=clipnorm)
        elif optimizer == 'adamax':
            optimizer_class = tf.keras.optimizers.Adamax(lr=self.lr, clipnorm=clipnorm)
        elif optimizer == 'adagrad':
            optimizer_class = tf.keras.optimizers.Adagrad(lr=self.lr, clipnorm=clipnorm)
        elif optimizer == 'adadelta':
            optimizer_class = tf.keras.optimizers.Adadelta(lr=self.lr, clipnorm=clipnorm)
        else:
            print("Optimizer not implemented yet")
            exit(1)

        # Timeline profiler
        if (self.conf is not None
                and conf['training']['timeline_prof']):
            self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self.run_metadata= tf.RunMetadata()
            self.model.compile(optimizer=optimizer_class, loss=loss,
                               options=self.run_options, run_metadata=self.run_metadata)
        else:
            self.model.compile(optimizer=optimizer_class, loss=loss)

        #self.ensure_equal_weights()

    def train_on_batch_and_get_deltas(self, X_batch, Y_batch, verbose=False):
        '''
        The purpose of the method is to perform a single gradient update over
        one mini-batch for one model replica.  Given a mini-batch, it first
        accesses the current model weights, performs single gradient update
        over one mini-batch, gets new model weights, calculates weight updates
        (deltas) by subtracting weight scalars, applies the learning rate.

        It performs calls to: subtract_params, multiply_params

        Argument list:
          - X_batch: input data for one mini-batch as a Numpy array
          - Y_batch: labels for one mini-batch as a Numpy array
          - verbose: set verbosity level (currently unused)

        Returns:
          - deltas: a list of model weight updates
          - loss: scalar training loss

        '''
        weights_before_update = self.model.get_weights()

        return_sequences = self.conf['model']['return_sequences']
        if not return_sequences:
            Y_batch = Y_batch[:, -1, :]
        loss = self.model.train_on_batch(X_batch, Y_batch)

        weights_after_update = self.model.get_weights()
        self.model.set_weights(weights_before_update)

        # unscale before subtracting
        weights_before_update = multiply_params(
            weights_before_update, 1.0/self.lr)
        weights_after_update = multiply_params(
            weights_after_update, 1.0/self.lr)

        deltas = subtract_params(weights_after_update, weights_before_update)

        # unscale loss
        if conf['model']['loss_scale_factor'] != 1.0:
            deltas = multiply_params(
                deltas, 1.0/conf['model']['loss_scale_factor'])

        return deltas, loss


    def build_callbacks(self, conf, callbacks_list):
        '''
        The purpose of the method is to set up logging and history. It is based
        on Keras Callbacks
        https://github.com/fchollet/keras/blob/fbc9a18f0abc5784607cd4a2a3886558efa3f794/keras/callbacks.py

        Currently used callbacks include: BaseLogger, CSVLogger, EarlyStopping.
        Other possible callbacks to add in future: RemoteMonitor,
        LearningRateScheduler

        Argument list:
        - conf: There is a "callbacks" section in conf.yaml file.

        Relevant parameters are:
        - list: Parameter specifying additional callbacks, read
        in the driver script and passed as an argument of type  list (see next
        arg)

        - metrics: List of quantities monitored during training and validation

        - mode: one of {auto, min, max}. The decision to overwrite the current
        save file is made based on either the maximization or the minimization
        of the monitored quantity. For val_acc, this should be max, for
        val_loss this should be min, etc. In auto mode, the direction is
        automatically inferred from the name of the monitored quantity.

        - monitor: Quantity used for early stopping, has to
        be from the list of metrics

        - patience: Number of epochs used to decide on whether to apply early
          stopping or continue training

        - callbacks_list: uses callbacks.list configuration parameter,
          specifies the list of additional callbacks Returns: modified list of
          callbacks

        '''

        mode = conf['callbacks']['mode']
        monitor = conf['callbacks']['monitor']
        patience = conf['callbacks']['patience']
        csvlog_save_path = conf['paths']['csvlog_save_path']
        # CSV callback is on by default
        if not os.path.exists(csvlog_save_path):
            os.makedirs(csvlog_save_path)

        callbacks_list = conf['callbacks']['list']
        callbacks = [tf.keras.callbacks.BaseLogger()]
        callbacks += [self.history]
        callbacks += [tf.keras.callbacks.CSVLogger("{}callbacks-{}.log".format(
            csvlog_save_path,
            datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))]

        if "earlystop" in callbacks_list:
            callbacks += [tf.keras.callbacks.EarlyStopping(
                patience=patience, monitor=monitor, mode=mode)]
        if "lr_scheduler" in callbacks_list:
            pass

        return tf.keras.callbacks.CallbackList(callbacks)

    def add_noise(self, X):
        if self.conf['training']['noise'] is True:
            prob = 0.05
        else:
            prob = self.conf['training']['noise']
        for i in range(0, X.shape[0]):
            for j in range(0, X.shape[2]):
                a = random.randint(0, 100)
                if a < prob*100:
                    X[i, :, j] = 0.0
        return X

    def train_epoch(self):
        '''
        Perform distributed mini-batch SGD for one epoch. It takes the batch
        iterator function and a model, fetches mini-batches in a while-loop
        until number of samples seen by the ensemble of workers (num_so_far)
        exceeds the training dataset size (num_total).

        NOTE: "sample" = "an entire shot" within this description

        During each iteration, the gradient updates (deltas) and the loss are
        calculated for each model replica in the ensemble, weights are averaged
        over ensemble, and the new weights are set.

        It performs calls to: MPIModel.get_deltas, MPIModel.set_new_weights
        methods

        Argument list: Empty

        Returns:
          - step: final iteration number
          - ave_loss: model loss averaged over iterations within this epoch
          - curr_loss: training loss averaged over replicas at final iteration
          - num_so_far: the cumulative number of samples seen by the ensemble
        of replicas up to the end of the final iteration (step) of this epoch

        Intermediate outputs and logging: debug printout of task_index (MPI),
        epoch number, number of samples seen to a current epoch, average
        training loss

        '''

        verbose = False
        first_run = True
        step = 0
        loss_averager = Averager()
        t_start = time.time()

        timeline_prof = False
        if (self.conf is not None
                and conf['training']['timeline_prof']):
            timeline_prof = True
        step_limit = 0
        if (self.conf is not None
                and conf['training']['step_limit'] > 0):
            step_limit = conf['training']['step_limit']

        batch_iterator_func = self.batch_iterator_func
        num_total = 1
        ave_loss = -1
        curr_loss = -1
        t0 = 0
        t1 = 0
        t2 = 0

        while ((self.num_so_far - self.epoch * num_total) < num_total
               or step < self.num_batches_minimum):
            if step_limit > 0 and step > step_limit:
                print('reached step limit')
                break
            try:
                (batch_xs, batch_ys, batches_to_reset, num_so_far_curr,
                 num_total, is_warmup_period) = next(batch_iterator_func)
            except StopIteration:
                g.print_unique("Resetting batch iterator.")
                self.num_so_far_accum = self.num_so_far_indiv
                self.set_batch_iterator_func()
                batch_iterator_func = self.batch_iterator_func
                (batch_xs, batch_ys, batches_to_reset, num_so_far_curr,
                 num_total, is_warmup_period) = next(batch_iterator_func)
            self.num_so_far_indiv = self.num_so_far_accum + num_so_far_curr

            # if batches_to_reset:
            # self.model.reset_states(batches_to_reset)

            warmup_phase = (step < self.warmup_steps and self.epoch == 0)
            num_replicas = 1 if warmup_phase else self.num_replicas

            self.num_so_far = self.mpi_sum_scalars(
                self.num_so_far_indiv, num_replicas)

            # run the model once to force compilation. Don't actually use these
            # values.
            if first_run:
                first_run = False
                t0_comp = time.time()
                #   print('input_dimension:',batch_xs.shape)
                #   print('output_dimension:',batch_ys.shape)
                _, _ = self.train_on_batch_and_get_deltas(
                    batch_xs, batch_ys, verbose)
                self.comm.Barrier()
                sys.stdout.flush()
                # TODO(KGF): check line feed/carriage returns around this
                g.print_unique('\nCompilation finished in {:.2f}s'.format(
                    time.time() - t0_comp))
                t_start = time.time()
                sys.stdout.flush()

            if np.any(batches_to_reset):
                reset_states(self.model, batches_to_reset)
            if ('noise' in self.conf['training'].keys()
                    and self.conf['training']['noise'] is not False):
                batch_xs = self.add_noise(batch_xs)
            t0 = time.time()
            deltas, loss = self.train_on_batch_and_get_deltas(
                batch_xs, batch_ys, verbose)
            t1 = time.time()
            if not is_warmup_period:
                self.set_new_weights(deltas, num_replicas)
                t2 = time.time()
                write_str_0 = self.calculate_speed(t0, t1, t2, num_replicas)
                curr_loss = self.mpi_average_scalars(1.0*loss, num_replicas)
                # g.print_unique(self.model.get_weights()[0][0][:4])
                loss_averager.add_val(curr_loss)
                ave_loss = loss_averager.get_ave()
                eta = self.estimate_remaining_time(
                    t0 - t_start, self.num_so_far - self.epoch*num_total,
                    num_total)
                write_str = (
                    '\r[{}] step: {} [ETA: {:.2f}s] [{:.2f}/{}], '.format(
                        self.task_index, step, eta, 1.0*self.num_so_far,
                        num_total)
                    + 'loss: {:.5f} [{:.5f}] | '.format(ave_loss, curr_loss)
                    + 'walltime: {:.4f} | '.format(
                        time.time() - self.start_time))
                g.write_unique(write_str + write_str_0)

                if timeline_prof:
                    # dump profile
                    tl = timeline.Timeline(self.run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    # dump file per iteration
                    with open('timeline_%s.json' % step, 'w') as f:
                        f.write(ctf)

                step += 1
            else:
                g.write_unique('\r[{}] warmup phase, num so far: {}'.format(
                    self.task_index, self.num_so_far))

        effective_epochs = 1.0*self.num_so_far/num_total
        epoch_previous = self.epoch
        self.epoch = effective_epochs
        g.write_unique(
            # TODO(KGF): "a total of X epochs within this session" ?
            '\nFinished training epoch {:.2f} '.format(self.epoch)
            # TODO(KGF): "precisely/exactly X epochs just passed"?
            + 'during this session ({:.2f} epochs passed)'.format(
                self.epoch - epoch_previous)
            # '\nEpoch {:.2f} finished training ({:.2f} epochs passed)'.format(
            #     1.0 * self.epoch, self.epoch - epoch_previous)
            + ' in {:.2f} seconds\n'.format(t2 - t_start))
        return (step, ave_loss, curr_loss, self.num_so_far, effective_epochs)

    def estimate_remaining_time(self, time_so_far, work_so_far, work_total):
        eps = 1e-6
        total_time = 1.0*time_so_far*work_total/(work_so_far + eps)
        return total_time - time_so_far

    def get_effective_lr(self, num_replicas):
        effective_lr = self.lr * num_replicas
        if effective_lr > self.max_lr:
            g.write_unique(
                'Warning: effective learning rate set to {}, '.format(
                    effective_lr)
                + 'larger than maximum {}. Clipping.'.format(self.max_lr))
            effective_lr = self.max_lr
        return effective_lr

    def get_effective_batch_size(self, num_replicas):
        return self.batch_size*num_replicas

    def calculate_speed(self, t0, t_after_deltas, t_after_update, num_replicas,
                        verbose=False):
        effective_batch_size = self.get_effective_batch_size(num_replicas)
        t_calculate = t_after_deltas - t0
        t_sync = t_after_update - t_after_deltas
        t_tot = t_after_update - t0

        examples_per_sec = effective_batch_size/t_tot
        frac_calculate = t_calculate/t_tot
        frac_sync = t_sync/t_tot

        print_str = ('{:.2E} Examples/sec | {:.2E} sec/batch '.format(
            examples_per_sec, t_tot)
                     + '[{:.1%} calc., {:.1%} sync.]'.format(
                         frac_calculate, frac_sync))
        print_str += '[batch = {} = {}*{}] [lr = {:.2E} = {:.2E}*{}]'.format(
            effective_batch_size, self.batch_size, num_replicas,
            self.get_effective_lr(num_replicas), self.lr, num_replicas)
        if verbose:
            g.write_unique(print_str)
        return print_str


    def mpi_make_predictions(conf, shot_list, loader, custom_path=None):
        loader.set_inference_mode(True)
        np.random.seed(g.task_index)
        shot_list.sort()  # make sure all replicas have the same list
        specific_builder = builder.ModelBuilder(conf)

        y_prime = []
        y_gold = []
        disruptive = []

        model = specific_builder.build_model(True)
        specific_builder.load_model_weights(model, custom_path)

        # broadcast model weights then set it explicitly: fix for Py3.6
        if g.task_index == 0:
            new_weights = model.get_weights()
        else:
            new_weights = None
        nw = g.comm.bcast(new_weights, root=0)
        model.set_weights(nw)

        model.reset_states()
        if g.task_index == 0:
            # TODO(KGF): this appears to prepend a \n, resulting in:
            # [2] loading from epoch 7
            #
            # 128/862 [===>..........................] - ETA: 2:20
            pbar = Progbar(len(shot_list))
        shot_sublists = shot_list.sublists(conf['model']['pred_batch_size'],
                                           do_shuffle=False, equal_size=True)
        y_prime_global = []
        y_gold_global = []
        disruptive_global = []
        if g.task_index != 0:
            loader.verbose = False

        for (i, shot_sublist) in enumerate(shot_sublists):
            if i % g.num_workers == g.task_index:
                X, y, shot_lengths, disr = loader.load_as_X_y_pred(shot_sublist)

                # load data and fit on data
                y_p = model.predict(X, batch_size=conf['model']['pred_batch_size'])
                model.reset_states()
                y_p = loader.batch_output_to_array(y_p)
                y = loader.batch_output_to_array(y)

                # cut arrays back
                y_p = [arr[:shot_lengths[j]] for (j, arr) in enumerate(y_p)]
                y = [arr[:shot_lengths[j]] for (j, arr) in enumerate(y)]

                y_prime += y_p
                y_gold += y
                disruptive += disr
                # print_all('\nFinished with i = {}'.format(i))

            if (i % g.num_workers == g.num_workers - 1
                    or i == len(shot_sublists) - 1):
                g.comm.Barrier()
                y_prime_global += concatenate_sublists(g.comm.allgather(y_prime))
                y_gold_global += concatenate_sublists(g.comm.allgather(y_gold))
                disruptive_global += concatenate_sublists(
                    g.comm.allgather(disruptive))
                g.comm.Barrier()
                y_prime = []
                y_gold = []
                disruptive = []

            if g.task_index == 0:
                pbar.add(1.0*len(shot_sublist))

        y_prime_global = y_prime_global[:len(shot_list)]
        y_gold_global = y_gold_global[:len(shot_list)]
        disruptive_global = disruptive_global[:len(shot_list)]
        loader.set_inference_mode(False)

        return y_prime_global, y_gold_global, disruptive_global


    def mpi_make_predictions_and_evaluate(conf, shot_list, loader,
                                          custom_path=None):
        y_prime, y_gold, disruptive = mpi_make_predictions(
            conf, shot_list, loader, custom_path)
        analyzer = PerformanceAnalyzer(conf=conf)
        roc_area = analyzer.get_roc_area(y_prime, y_gold, disruptive)
        shot_list.set_weights(
            analyzer.get_shot_difficulty(y_prime, y_gold, disruptive))
        loss = get_loss_from_list(y_prime, y_gold, conf['data']['target'])
        return y_prime, y_gold, disruptive, roc_area, loss


    def mpi_make_predictions_and_evaluate_multiple_times(conf, shot_list, loader,
                                                         times, custom_path=None):
        y_prime, y_gold, disruptive = mpi_make_predictions(conf, shot_list, loader,
                                                           custom_path)
        areas = []
        losses = []
        for T_min_curr in times:
            # if 'monitor_test' in conf['callbacks'].keys() and
            # conf['callbacks']['monitor_test']:
            conf_curr = deepcopy(conf)
            T_min_warn_orig = conf['data']['T_min_warn']
            conf_curr['data']['T_min_warn'] = T_min_curr
            assert conf['data']['T_min_warn'] == T_min_warn_orig
            analyzer = PerformanceAnalyzer(conf=conf_curr)
            roc_area = analyzer.get_roc_area(y_prime, y_gold, disruptive)
            # shot_list.set_weights(analyzer.get_shot_difficulty(y_prime, y_gold,
            # disruptive))
            loss = get_loss_from_list(y_prime, y_gold, conf['data']['target'])
            areas.append(roc_area)
            losses.append(loss)
        return areas, losses


    def train(conf, shot_list_train, shot_list_validate, loader,
              callbacks_list=None, shot_list_test=None):

        loader.set_inference_mode(False)

        # TODO(KGF): this is not defined in conf.yaml, but added to processed dict
        # for the first time here:
        conf['num_workers'] = g.comm.Get_size()

        specific_builder = builder.ModelBuilder(conf)
        if g.tf_ver >= parse_version('1.14.0'):
            # Internal TensorFlow flags, subject to change (v1.14.0+ only?)
            try:
                from tensorflow.python.util import module_wrapper as depr
            except ImportError:
                from tensorflow.python.util import deprecation_wrapper as depr
            # depr._PRINT_DEPRECATION_WARNINGS = False  # does nothing
            depr._PER_MODULE_WARNING_LIMIT = 0
            # Suppresses warnings from "keras/backend/tensorflow_backend.py"
            # except: "Rate should be set to `rate = 1 - keep_prob`"
            # Also suppresses warnings from "keras/optimizers.py
            # does NOT suppresses warn from "/tensorflow/python/ops/math_grad.py"
        else:
            # TODO(KGF): next line suppresses ALL info and warning messages,
            # not just deprecation warnings...
            tf.logging.set_verbosity(tf.logging.ERROR)
        # TODO(KGF): for TF>v1.13.0 (esp v1.14.0), this next line prompts a ton of
        # deprecation warnings with externally-packaged Keras, e.g.:
        # WARNING:tensorflow:From  .../keras/backend/tensorflow_backend.py:174:
        # The name tf.get_default_session is deprecated.
        # Please use tf.compat.v1.get_default_session instead.
        train_model = specific_builder.build_model(False)
        # Cannot fix these Keras internals via "import tensorflow.compat.v1 as tf"
        #
        # TODO(KGF): note, these are different than C-based info diagnostics e.g.:
        # 2019-11-06 18:27:31.698908: I ...  dynamic library libcublas.so.10
        # which are NOT suppressed by set_verbosity. See top level __init__.py

        # load the latest epoch we did. Returns 0 if none exist yet
        e = specific_builder.load_model_weights(train_model)
        e_old = e

        num_epochs = conf['training']['num_epochs']
        lr_decay = conf['model']['lr_decay']
        batch_size = conf['training']['batch_size']
        lr = conf['model']['lr']
        clipnorm = conf['model']['clipnorm']
        warmup_steps = conf['model']['warmup_steps']
        # TODO(KGF): rename as "num_iter_minimum" or "min_steps_per_epoch"
        num_batches_minimum = conf['training']['num_batches_minimum']

        if 'adam' in conf['model']['optimizer']:
            optimizer = MPIAdam(lr=lr)
        elif (conf['model']['optimizer'] == 'sgd'):
            optimizer = MPISGD(lr=lr)
        elif 'momentum_sgd' in conf['model']['optimizer']:
            optimizer = MPIMomentumSGD(lr=lr)
        else:
            print("Optimizer not implemented yet")
            exit(1)

        g.print_unique('{} epoch(s) left to go'.format(num_epochs - e))

        batch_generator = partial(loader.training_batch_generator_partial_reset,
                                  shot_list=shot_list_train)

        g.print_unique("warmup steps = {}".format(warmup_steps))
        mpi_model = MPIModel(train_model, optimizer, g.comm, batch_generator,
                             batch_size, lr=lr, warmup_steps=warmup_steps,
                             num_batches_minimum=num_batches_minimum, conf=conf)
        mpi_model.compile(conf['model']['optimizer'], clipnorm,
                          conf['data']['target'].loss)
        tensorboard = None
        if g.task_index == 0:
            tensorboard_save_path = conf['paths']['tensorboard_save_path']
            write_grads = conf['callbacks']['write_grads']
            tensorboard = TensorBoard(log_dir=tensorboard_save_path,
                                      histogram_freq=1, # write_graph=True,
                                      write_grads=write_grads)
            tensorboard.set_model(mpi_model.model)
            # TODO(KGF): check addition of TF model summary write added from fork
            fr = open('model_architecture.log', 'a')
            ori = sys.stdout
            sys.stdout = fr
            mpi_model.model.summary()
            sys.stdout = ori
            fr.close()
            mpi_model.model.summary()

        if g.task_index == 0:
            callbacks = mpi_model.build_callbacks(conf, callbacks_list)
            callbacks.set_model(mpi_model.model)
            callback_metrics = conf['callbacks']['metrics']
            callbacks.set_params({'epochs': num_epochs,
                                  'metrics': callback_metrics,
                                  'batch_size': batch_size, })
            callbacks.on_train_begin()
        if conf['callbacks']['mode'] == 'max':
            best_so_far = -np.inf
            cmp_fn = max
        else:
            best_so_far = np.inf
            cmp_fn = min

        while e < num_epochs:
            g.write_unique('\nBegin training from epoch {:.2f}/{}'.format(
                e, num_epochs))
            if g.task_index == 0:
                callbacks.on_epoch_begin(int(round(e)))
            mpi_model.set_lr(lr*lr_decay**e)

            # KGF: core work of loop performed in next line
            (step, ave_loss, curr_loss, num_so_far,
             effective_epochs) = mpi_model.train_epoch()
            e = e_old + effective_epochs
            g.write_unique('Finished training of epoch {:.2f}/{}\n'.format(
                e, num_epochs))

            # TODO(KGF): add diagnostic about "saving to epoch X"?
            loader.verbose = False  # True during the first iteration
            if g.task_index == 0:
                specific_builder.save_model_weights(train_model, int(round(e)))

            if conf['training']['no_validation']:
                break

            epoch_logs = {}
            g.write_unique('Begin evaluation of epoch {:.2f}/{}\n'.format(
                e, num_epochs))
            # TODO(KGF): flush output/ MPI barrier?
            # g.flush_all_inorder()

            # TODO(KGF): is there a way to avoid Keras.Models.load_weights()
            # repeated calls throughout mpi_make_pred*() fn calls?
            _, _, _, roc_area, loss = mpi_make_predictions_and_evaluate(
                conf, shot_list_validate, loader)

            if conf['training']['ranking_difficulty_fac'] != 1.0:
                (_, _, _, roc_area_train,
                 loss_train) = mpi_make_predictions_and_evaluate(
                     conf, shot_list_train, loader)
                batch_generator = partial(
                    loader.training_batch_generator_partial_reset,
                    shot_list=shot_list_train)
                mpi_model.batch_iterator = batch_generator
                mpi_model.batch_iterator_func.__exit__()
                mpi_model.num_so_far_accum = mpi_model.num_so_far_indiv
                mpi_model.set_batch_iterator_func()

            if ('monitor_test' in conf['callbacks'].keys()
                    and conf['callbacks']['monitor_test']):
                times = conf['callbacks']['monitor_times']
                areas, _ = mpi_make_predictions_and_evaluate_multiple_times(
                    conf, shot_list_validate, loader, times)
                epoch_str = 'epoch {}, '.format(int(round(e)))
                g.write_unique(epoch_str + ' '.join(
                    ['val_roc_{} = {}'.format(t, roc) for t, roc in zip(
                        times, areas)]
                    ) + '\n')
                if shot_list_test is not None:
                    areas, _ = mpi_make_predictions_and_evaluate_multiple_times(
                        conf, shot_list_test, loader, times)
                    g.write_unique(epoch_str + ' '.join(
                        ['test_roc_{} = {}'.format(t, roc) for t, roc in zip(
                            times, areas)]
                        ) + '\n')

            epoch_logs['val_roc'] = roc_area
            epoch_logs['val_loss'] = loss
            epoch_logs['train_loss'] = ave_loss
            best_so_far = cmp_fn(epoch_logs[conf['callbacks']['monitor']],
                                 best_so_far)
            stop_training = False
            g.flush_all_inorder()
            if g.task_index == 0:
                print('=========Summary======== for epoch {:.2f}'.format(e))
                print('Training Loss numpy: {:.3e}'.format(ave_loss))
                print('Validation Loss: {:.3e}'.format(loss))
                print('Validation ROC: {:.4f}'.format(roc_area))
                if conf['training']['ranking_difficulty_fac'] != 1.0:
                    print('Training Loss: {:.3e}'.format(loss_train))
                    print('Training ROC: {:.4f}'.format(roc_area_train))
                print('======================== ')
                callbacks.on_epoch_end(int(round(e)), epoch_logs)
                if hasattr(mpi_model.model, 'stop_training'):
                    stop_training = mpi_model.model.stop_training
                # only save model weights if quantity we are tracking is improving
                if best_so_far != epoch_logs[conf['callbacks']['monitor']]:
                    if ('monitor_test' in conf['callbacks'].keys()
                            and conf['callbacks']['monitor_test']):
                        print("No improvement, saving model weights anyways")
                    else:
                        print("Not saving model weights")
                        specific_builder.delete_model_weights(
                            train_model, int(round(e)))

                # tensorboard
                val_generator = partial(loader.training_batch_generator,
                                        shot_list=shot_list_validate)()
                val_steps = 1
                tensorboard.on_epoch_end(val_generator, val_steps,
                                         int(round(e)), epoch_logs)
            stop_training = g.comm.bcast(stop_training, root=0)
            g.write_unique('Finished evaluation of epoch {:.2f}/{}'.format(
                e, num_epochs))
            # TODO(KGF): compare to old diagnostic:
            # g.write_unique("end epoch {}".format(e_old))
            if stop_training:
                g.write_unique("Stopping training due to early stopping")
                break

        if g.task_index == 0:
            callbacks.on_train_end()
            tensorboard.on_train_end()

        mpi_model.close()


    def get_stop_training(callbacks):
        # TODO(KGF): this funciton is unused
        for cb in callbacks.callbacks:
            if isinstance(cb, tf.keras.callbacks.EarlyStopping):
                print("Checking for early stopping")
                return cb.model.stop_training
        print("No early stopping callback found.")
        return False


class TensorBoard(object):
    def __init__(self, log_dir='./logs', histogram_freq=0, validation_steps=0,
                 # write_graph=True,
                 write_grads=False):
        if K.backend() != 'tensorflow':
            raise RuntimeError('TensorBoard callback only works '
                               'with the TensorFlow backend.')
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.writer = None
        # self.write_graph = write_graph
        self.write_grads = write_grads
        self.validation_steps = validation_steps
        self.model = None

    def set_model(self, model):
        self.model = model

        # TODO(KGF): check removal of cond "&& self.merged is None"
        if self.histogram_freq:
            for layer in self.model.layers:
                for weight in layer.weights:
                    mapped_weight_name = weight.name.replace(':', '_')
                    tf.summary.histogram(mapped_weight_name, weight)
                    if self.write_grads:
                        grads = self.model.optimizer.get_gradients(
                            self.model.total_loss, weight)

                        def is_indexed_slices(grad):
                            return type(grad).__name__ == 'IndexedSlices'
                        grads = [grad.values if is_indexed_slices(grad) else
                                 grad for grad in grads]
                        for grad in grads:
                            tf.summary.histogram(
                                '{}_grad'.format(mapped_weight_name), grad)

                # KGF: Skip writing layer output histograms in TF 2.x, for now?
                if hasattr(layer, 'output'):
                    tf.summary.histogram('{}_out'.format(layer.name),
                                         layer.output)

        self.writer = tf.summary.create_file_writer(self.log_dir)

    def on_epoch_end(self, val_generator, val_steps, epoch, logs=None):
        logs = logs or {}

        # KGF: val_roc, val_loss, train_loss
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            tf.summary.scalar(name, value, step=epoch)
            self.writer.flush()

    def on_train_end(self):
        self.writer.close()


class ModelBuilder(object):
    def __init__(self, conf):
        self.conf = conf

    def get_unique_id(self):
        this_conf = deepcopy(self.conf)
        # ignore hash depednecy on number of epochs or T_min_warn (they are
        # both modifiable). Map local copy of all confs to the same values
        this_conf['training']['num_epochs'] = 0
        this_conf['data']['T_min_warn'] = 30
        unique_id = general_object_hash(this_conf)
        return unique_id

    def get_0D_1D_indices(self):
        # make sure all 1D indices are contiguous in the end!
        use_signals = self.conf['paths']['use_signals']
        indices_0d = []
        indices_1d = []
        num_0D = 0
        num_1D = 0
        curr_idx = 0
        # do we have any 1D indices?
        is_1D_region = use_signals[0].num_channels > 1
        for sig in use_signals:
            num_channels = sig.num_channels
            indices = range(curr_idx, curr_idx+num_channels)
            if num_channels > 1:
                indices_1d += indices
                num_1D += 1
                is_1D_region = True
            else:
                assert not is_1D_region, (
                    "Check that use_signals are ordered with 1D signals last!")
                assert num_channels == 1
                indices_0d += indices
                num_0D += 1
                is_1D_region = False
            curr_idx += num_channels
        return (np.array(indices_0d).astype(np.int32),
                np.array(indices_1d).astype(np.int32), num_0D, num_1D)

    def save_model_weights(self, model, epoch):
        # Keras HDF5 weights only
        save_path = self.get_save_path(epoch)
        model.save_weights(save_path, overwrite=True)
        # Keras light-weight HDF5 format: model arch, weights, compile info
        full_model_save_path = self.get_save_path(epoch, ext='hdf5')
        model.save(full_model_save_path,
                   overwrite=True,  # default
                   include_optimizer=True,  # default
                   save_format=None,  # default, 'h5' in r1.15. Else 'tf'
                   signatures=None,  # applicable to 'tf' SavedModel format only
                   )
        # TensorFlow SavedModel format (full directory)
        full_model_save_dir = full_model_save_path.rsplit('.', 1)[0]
        # TODO(KGF): model.save(..., save_format='tf') disabled in r1.15
        # Same with tf.keras.models.save_model(..., save_format="tf").
        # Need to use experimental API until r2.x
        model.save(full_model_save_dir, overwrite=True, save_format='tf')

        # try:
        if _has_onnx:
            save_path = self.get_save_path(epoch, ext='onnx')
            onnx_model = keras2onnx.convert_keras(model, model.name,
                                                  target_opset=10)
            onnx.save_model(onnx_model, save_path)
        # except Exception as e:
        #     print(e)
        return

    def delete_model_weights(self, model, epoch):
        save_path = self.get_save_path(epoch)
        assert os.path.exists(save_path)
        os.remove(save_path)

    def get_save_path(self, epoch, ext='h5'):
        unique_id = self.get_unique_id()
        dir_path = self.conf['paths']['model_save_path']
        # TODO(KGF): consider storing .onnx files in subdirectory away from .h5
        # if ext == 'onnx':
        #     os.path.join(dir_path, 'onnx/')
        return os.path.join(
            dir_path, 'model.{}._epoch_.{}.{}'.format(unique_id, epoch, ext))

    def ensure_save_directory(self):
        prepath = self.conf['paths']['model_save_path']
        makedirs_process_safe(prepath)

    def load_model_weights(self, model, custom_path=None):
        if custom_path is None:
            epochs = self.get_all_saved_files()
            if len(epochs) == 0:
                g.write_all('no previous checkpoint found\n')
                # TODO(KGF): port indexing change (from "return -1") to parts
                # of the code other than mpi_runner.py
                return 0
            else:
                max_epoch = max(epochs)
                g.write_all('loading from epoch {}\n'.format(max_epoch))
                model.load_weights(self.get_save_path(max_epoch))
                return max_epoch
        else:
            epoch = self.extract_id_and_epoch_from_filename(
                os.path.basename(custom_path))[1]
            model.load_weights(custom_path)
            g.write_all("Loading from custom epoch {}\n".format(epoch))
            return epoch


    def extract_id_and_epoch_from_filename(self, filename):
        regex = re.compile(r'-?\d+')
        numbers = [int(x) for x in regex.findall(filename)]
        # TODO: should ignore any files that dont match our naming convention
        # in this directory, especially since we are now writing full .hdf5 too.
        # Will crash the program if, e.g., a .tgz file is in that directory
        if filename[-3:] == '.h5':
            assert len(numbers) == 3  # id, epoch number, and .h5 extension
            assert numbers[2] == 5  # .h5 extension
        return numbers[0], numbers[1]

    def get_all_saved_files(self):
        self.ensure_save_directory()
        unique_id = self.get_unique_id()
        path = self.conf['paths']['model_save_path']
        # TODO(KGF): probably should only list .h5 file, not ONNX right now
        filenames = [name for name in os.listdir(path)
                     if os.path.isfile(os.path.join(path, name))]
        epochs = []
        for fname in filenames:
            curr_id, epoch = self.extract_id_and_epoch_from_filename(fname)
            if curr_id == unique_id:
                epochs.append(epoch)
        return epochs


    from copy import deepcopy
from src.utils.hashing import general_object_hash
from src.models.tensorflow.tcn import TCN
# TODO(KGF): consider using importlib.util.find_spec() instead (Py>3.4)
try:
    import keras2onnx
    import onnx
except ImportError:  # as e:
    _has_onnx = False
    # onnx = None
    # keras2onnx = None
else:
    _has_onnx = True

# Synchronize 2x stderr msg from TensorFlow initialization via Keras backend
# "Succesfully opened dynamic library... libcudart" "Using TensorFlow backend."
if g.comm is not None:
    g.flush_all_inorder()


def build_model(self, predict):
    conf = self.conf
    model_conf = conf['model']
    rnn_size = model_conf['rnn_size']
    rnn_type = model_conf['rnn_type']
    regularization = model_conf['regularization']
    dense_regularization = model_conf['dense_regularization']

    dropout_prob = model_conf['dropout_prob']
    length = model_conf['length']
    pred_length = model_conf['pred_length']
    # skip = model_conf['skip']
    stateful = model_conf['stateful']
    return_sequences = model_conf['return_sequences']
    # model_conf['output_activation']
    output_activation = conf['data']['target'].activation
    use_signals = conf['paths']['use_signals']
    num_signals = sum([sig.num_channels for sig in use_signals])
    num_conv_filters = model_conf['num_conv_filters']
    # num_conv_layers = model_conf['num_conv_layers']
    size_conv_filters = model_conf['size_conv_filters']
    pool_size = model_conf['pool_size']
    dense_size = model_conf['dense_size']

    batch_size = self.conf['training']['batch_size']
    if predict:
        batch_size = self.conf['model']['pred_batch_size']
        # so we can predict with one time point at a time!
        if return_sequences:
            length = pred_length
        else:
            length = 1

    if rnn_type == 'LSTM':
        rnn_model = LSTM
    elif rnn_type == 'SimpleRNN':
        rnn_model = SimpleRNN
    else:
        print('Unkown Model Type, exiting.')
        exit(1)

    batch_input_shape = (batch_size, length, num_signals)

    indices_0d, indices_1d, num_0D, num_1D = self.get_0D_1D_indices()

    # def slicer(x, indices):
    #     return x[:, indices]

    # def slicer_output_shape(input_shape, indices):
    #     shape_curr = list(input_shape)
    #     assert len(shape_curr) == 2  # only valid for 3D tensors
    #     shape_curr[-1] = len(indices)
    #     return tuple(shape_curr)

    pre_rnn_input = Input(shape=(num_signals,))

    # if num_1D > 0:

    # else:
    #     pre_rnn = pre_rnn_input


    pre_rnn_model = tf.keras.Model(inputs=pre_rnn_input, outputs=pre_rnn)
    x_input = Input(batch_shape=batch_input_shape)
    # TODO(KGF): Ge moved this inside a new conditional in Dec 2019. check
    # x_in = TimeDistributed(pre_rnn_model)(x_input)
    if num_1D > 0:
        x_in = TimeDistributed(pre_rnn_model)(x_input)
    else:
        x_in = x_input

    # ==========
    # TCN MODEL
    # ==========
    if ('keras_tcn' in model_conf.keys()
            and model_conf['keras_tcn'] is True):
        print('Building TCN model....')
        tcn_layers = model_conf['tcn_layers']
        tcn_dropout = model_conf['tcn_dropout']
        nb_filters = model_conf['tcn_hidden']
        kernel_size = model_conf['kernel_size_temporal']
        nb_stacks = model_conf['tcn_nbstacks']
        use_skip_connections = model_conf['tcn_skip_connect']
        activation = model_conf['tcn_activation']
        use_batch_norm = model_conf['tcn_batch_norm']
        for _ in range(model_conf['tcn_pack_layers']):
            x_in = TCN(
                use_batch_norm=use_batch_norm, activation=activation,
                use_skip_connections=use_skip_connections,
                nb_stacks=nb_stacks, kernel_size=kernel_size,
                nb_filters=nb_filters, num_layers=tcn_layers,
                dropout_rate=tcn_dropout)(x_in)
            x_in = Dropout(dropout_prob)(x_in)
    else:  # end TCN model
        # ==========
        # RNN MODEL
        # ==========
        # LSTM in ONNX: "The maximum opset needed by this model is only 9."
        rnn_kwargs = dict(return_sequences=return_sequences,
                          # batch_input_shape=batch_input_shape,
                          stateful=stateful,
                          kernel_regularizer=l2(regularization),
                          recurrent_regularizer=l2(regularization),
                          bias_regularizer=l2(regularization),
                          dropout=dropout_prob,
                          )
        # https://stackoverflow.com/questions/60468385/is-there-cudnnlstm-or-cudnngru-alternative-in-tensorflow-2-0
        # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/keras/layers/recurrent_v2.py#L902
        if rnn_type != 'CuDNNLSTM':
            # Recurrent Dropout is unsupported in CuDNN library
            rnn_kwargs['recurrent_dropout'] = dropout_prob
        for _ in range(model_conf['rnn_layers']):
            x_in = rnn_model(rnn_size, **rnn_kwargs)(x_in)
            # KGF: is this redundant with the dropout within the LSTM cell?
            x_in = Dropout(dropout_prob)(x_in)
        if return_sequences:
            x_out = TimeDistributed(
                Dense(1, activation=output_activation))(x_in)
    model = tf.keras.Model(inputs=x_input, outputs=x_out)
    model.reset_states()
    return model


 class LossHistory(Callback):
    def on_train_begin(self, logs=None):
        self.losses = []

    def on_batch_end(self, batch, logs=None):
        self.losses.append(logs.get('loss'))
