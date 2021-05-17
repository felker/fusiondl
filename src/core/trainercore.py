from abc import ABC, abstractmethod
import os
import sys
import time
import tempfile
import copy
import numpy
import datetime
import pathlib

from src.core.processing import makedirs_process_safe

from collections import OrderedDict

# KGF: this class has about half as many methods as the derived TF class

# also, distributed_trainer is a child class of torch_trainer or tf_trainer
# so in both PyTorch and TF, there are only a few overridden class methods

# _initialize_io, batch_process, lr schedule, build_lr_schedule, log
# are the only functions in the base class doing any heavy lifting

# TODO(KGF): consider moving this from core/ to trainers/

class trainercore(ABC):
    '''
    This class is the core interface for training.

    CosmicTagger: "Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error." (not actually true, they just use "pass")

    For Python 3.4+, should instead derive class from abc.ABC and mark appropriate methods
    with @abstractmethod decorator
    '''


    def __init__(self, conf):
        self.conf = conf
        self.start_time = time.time()
        self.epoch = 0
        self.num_so_far = 0
        self.num_so_far_accum = 0
        self.num_so_far_indiv = 0
        self.max_lr = 0.1
        self.lr = self.max_lr

        # if args.data.data_format == "channels_first": self._channels_dim = 1
        # if args.data.data_format == "channels_last" : self._channels_dim = -1

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

    def print(self, *argv):
        ''' Function for logging as needed.  Works correctly in distributed mode'''

        message = " ".join([ str(s) for s in argv] )

        sys.stdout.write(message + "\n")
        sys.stdout.flush()

    @abstractmethod
    def initialize(self):
        # TODO(KGF): replace abstractmethod bodies from "return" to docstring
        # with implicit "return None"
        return

    def build_lr_schedule(self, learning_rate_schedule = None):
        # Define the learning rate sequence:

        if learning_rate_schedule is None:
            learning_rate_schedule = {
                'warm_up' : {
                    'function'      : 'linear',
                    'start'         : 0,
                    'n_epochs'      : 1,
                    'initial_rate'  : 0.00001,
                },
                'flat' : {
                    'function'      : 'flat',
                    'start'         : 1,
                    'n_epochs'      : 5,
                },
                'decay' : {
                    'function'      : 'decay',
                    'start'         : 6,
                    'n_epochs'      : 4,
                        'floor'         : 0.00001,
                    'decay_rate'    : 0.999
                },
            }


        # We build up the functions we need piecewise:
        func_list = []
        cond_list = []

        for i, key in enumerate(learning_rate_schedule):

            # First, create the condition for this stage
            start    = learning_rate_schedule[key]['start']
            length   = learning_rate_schedule[key]['n_epochs']

            if i +1 == len(learning_rate_schedule):
                # Make sure the condition is open ended if this is the last stage
                condition = lambda x, s=start, l=length: x >= s
            else:
                # otherwise bounded
                condition = lambda x, s=start, l=length: x >= s and x < s + l


            if learning_rate_schedule[key]['function'] == 'linear':

                initial_rate = learning_rate_schedule[key]['initial_rate']
                if 'final_rate' in learning_rate_schedule[key]: final_rate = learning_rate_schedule[key]['final_rate']
                else: final_rate = self.args.mode.optimizer.learning_rate

                function = lambda x, s=start, l=length, i=initial_rate, f=final_rate : numpy.interp(x, [s, s + l] ,[i, f] )

            elif learning_rate_schedule[key]['function'] == 'flat':
                if 'rate' in learning_rate_schedule[key]: rate = learning_rate_schedule[key]['rate']
                else: rate = self.args.mode.optimizer.learning_rate

                function = lambda x : rate

            elif learning_rate_schedule[key]['function'] == 'decay':
                decay    = learning_rate_schedule[key]['decay_rate']
                floor    = learning_rate_schedule[key]['floor']
                if 'rate' in learning_rate_schedule[key]: rate = learning_rate_schedule[key]['rate']
                else: rate = self.args.mode.optimizer.learning_rate

                function = lambda x, s=start, d=decay, f=floor: (rate-f) * numpy.exp( -(d * (x - s))) + f

            cond_list.append(condition)
            func_list.append(function)

        self.lr_calculator = lambda x: numpy.piecewise(
            x * (self.args.run.minibatch_size / self._train_data_size),
            [c(x * (self.args.run.minibatch_size / self._train_data_size)) for c in cond_list], func_list)

    @abstractmethod
    def build_model(self):
        return

    @abstractmethod
    def print_model_info(self):
        return

    @abstractmethod
    def set_compute_parameters(self):
        return


    def log(self, metrics, kind, step):

        log_string = ""

        log_string += "{} Global Step {}: ".format(kind, step)


        for key in metrics:
            if key in self._log_keys and key != "global_step":
                log_string += "{}: {:.3}, ".format(key, metrics[key])

        if kind == "Train":
            log_string += "Img/s: {:.2} ".format(metrics["images_per_second"])
            log_string += "IO: {:.2} ".format(metrics["io_fetch_time"])
        else:
            log_string.rstrip(", ")

        self.log(log_string)

        return

    @abstractmethod
    def on_step_end(self):
        return

    @abstractmethod
    def on_epoch_end(self):
        return

    def metrics(self, metrics):
        # This function looks useless, but it is not.
        # It allows a handle to the distributed network to allreduce metrics.
        return metrics

    @abstractmethod
    def stop(self):
        # Mostly, this is just turning off the io:
        # self._larcv_interface.stop()
        return

    @abstractmethod
    def close_savers(self):
        return

    @abstractmethod
    def save_model_weights(self, model, epoch):
        return

    @abstractmethod
    def get_save_path(self, epoch, ext='h5'):
        return

    @abstractmethod
    def extract_id_and_epoch_from_filename(self, filename):
        return

    def delete_model_weights(self, model, epoch):
        save_path = self.get_save_path(epoch)
        assert os.path.exists(save_path)
        os.remove(save_path)

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


    def batch_process(self):
        # KGF: main training loop, called in exec.py

        for self._iteration in range(self.args.run.iterations):
            if self.args.mode.name == "train" and self._iteration >= self.args.run.iterations:
                self.print('Finished training (iteration %d)' % self._iteration)
                self.checkpoint()
                break


            if self.args.mode.name == "train":
                self.val_step()
                self.train_step()
                self.checkpoint()
            else:
                self.ana_step()

        self.close_savers()

    def get_unique_id(self):
        ''' Hash nearly the entire conf.yaml for a unique model ID for writing results
        (not the same as the unique hash ID for input signals, etc., which barely depends
        on conf.yaml)
        '''

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
        # KGF: above requires that conf processor already has changed the default empty
        # list of "use_signals" to include all the valid ones
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


    def estimate_remaining_time(self, time_so_far, work_so_far, work_total):
        eps = 1e-6
        total_time = 1.0*time_so_far*work_total/(work_so_far + eps)
        return total_time - time_so_far

    # def get_effective_lr(self, num_replicas):
    #     effective_lr = self.lr * num_replicas
    #     if effective_lr > self.max_lr:
    #         g.write_unique(
    #             'Warning: effective learning rate set to {}, '.format(
    #                 effective_lr)
    #             + 'larger than maximum {}. Clipping.'.format(self.max_lr))
    #         effective_lr = self.max_lr
    #     return effective_lr

    # def get_effective_batch_size(self, num_replicas):
    #     return self.batch_size*num_replicas

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
