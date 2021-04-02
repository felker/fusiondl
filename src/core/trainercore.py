import os
import sys
import time
import tempfile
import copy

from collections import OrderedDict

import numpy

import datetime
import pathlib

# KGF: this class has about half as many methods as the derived TF class

# also, distributed_trainer is a child class of torch_trainer or tf_trainer
# so in both PyTorch and TF, there are only a few overridden class methods

# _initialize_io, batch_process, lr schedule, build_lr_schedule, log
# are the only functions in the base class doing any heavy lifting


class trainercore(object):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''


    def __init__(self, args):

        self._iteration    = 0
        self._global_step  = 0
        self.args          = args

        if args.data.data_format == "channels_first": self._channels_dim = 1
        if args.data.data_format == "channels_last" : self._channels_dim = -1


    def print(self, *argv):
        ''' Function for logging as needed.  Works correctly in distributed mode'''

        message = " ".join([ str(s) for s in argv] )

        sys.stdout.write(message + "\n")
        sys.stdout.flush()

    def initialize(self):
        pass

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


    def init_network(self):
        pass

    def print_network_info(self):
        pass

    def set_compute_parameters(self):
        pass


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

    def on_step_end(self):
        pass

    def on_epoch_end(self):
        pass

    def metrics(self, metrics):
        # This function looks useless, but it is not.
        # It allows a handle to the distributed network to allreduce metrics.
        return metrics

    def stop(self):
        # Mostly, this is just turning off the io:
        # self._larcv_interface.stop()
        pass

    def close_savers(self):
        pass

    def batch_process(self):

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


def makedirs_process_safe(dirpath):
    try:  # can lead to race condition
        os.makedirs(dirpath)
    except OSError as e:
        # File exists, and it's a directory, another process beat us to
        # creating this dir, that's OK.
        if e.errno == errno.EEXIST:
            pass
        else:
            # Our target dir exists as a file, or different error, reraise the
            # error!
            raise
