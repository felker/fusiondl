#!/usr/bin/env python

import os.path
import sys
import logging
import numpy as np
import argparse
import errno
# import hydra
# TODO(KGF): setup logging. Hydra will setup logging for project, which would break my
# setup

exec_dir = os.path.dirname(os.path.abspath(__file__))
proj_dir = os.path.dirname(exec_dir)
sys.path.insert(0, proj_dir)

# TODO(KGF): global_vars stores MPI communicator, conf dictionary, and fns for clean
# parallel stdout. Probably better alternative is to use large "class exec():" with
# self.args, etc. and explicitly pass around the conf to make_trainer() etc.
import src.global_vars as g
# only places CosmicTagger uses MPI outside Horovod is for logging in exec.py
# and torch/distributed_trainer.py iff DDP
g.init_MPI()

from src.config.conf_parser import read_parameters

# set PRNG seed, unique for each worker, based on MPI task index for
# reproducible shuffling in guranteed_preprocessed() and training steps
np.random.seed(g.task_index)
# random.seed(g.task_index)


def is_valid_file(parser, arg):
    if not (os.path.exists(arg) and os.path.isfile(arg)):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg


def make_parser():
    parser = argparse.ArgumentParser(prog='fusiondl', description='FusionDL suite')
    parser.add_argument("--input_file", "-i",   # type=str,
                        required=False, dest="conf_file",
                        help="input YAML file for configuration", metavar="YAML_FILE",
                        type=lambda x: is_valid_file(parser, x))

    parser.add_argument('-d','--distributed',
                        action  = 'store_true',
                        default = False,
                        help    = "Run with data parallelism")

    # parser.add_argument('--distributed-mode',
    #                     type    = str,
    #                     default = 'horovod',
    #                     choices = ['horovod', 'DDP'],
    #                     help    = "Toggle between the different methods for distributing
    #                     the network")

    # TODO(KGF): re-add only_predict option as in original code (or "iotest" config.mode
    # as in CosmicTagger hydra branch)

    return parser




## def init_mpi():


# def normalize(conf):
#     if conf['data']['normalizer'] == 'minmax':
#         from src.preprocessor.normalize import MinMaxNormalizer as Normalizer
#     elif conf['data']['normalizer'] == 'meanvar':
#         from src.preprocessor.normalize import MeanVarNormalizer as Normalizer
#     elif conf['data']['normalizer'] == 'var':
#         # performs !much better than minmaxnormalizer
#         from src.preprocessor.normalize import VarNormalizer as Normalizer
#     elif conf['data']['normalizer'] == 'averagevar':
#         # performs !much better than minmaxnormalizer
#         from src.preprocessor.normalize import (
#             AveragingVarNormalizer as Normalizer
#         )
#     else:
#         print('unkown normalizer. exiting')
#         exit(1)
#     normalizer = Normalizer(conf)
#     if g.task_index == 0:
#         # make sure preprocessing has been run, and results are saved to files
#         # if not, only master MPI rank spawns thread pool to perform preprocessing
#         (shot_list_train, shot_list_validate,
#          shot_list_test) = guarantee_preprocessed(conf)
#         # similarly, train normalizer (if necessary) w/ master MPI rank only
#         normalizer.train()  # verbose=False only suppresses if purely loading
#         g.comm.Barrier()
#         g.print_unique("begin preprocessor+normalization (all MPI ranks)...")
#         # second call has ALL MPI ranks load preprocessed shots from .npz files
#         (shot_list_train, shot_list_validate,
#          shot_list_test) = guarantee_preprocessed(conf, verbose=True)
#         # second call to normalizer training
#         normalizer.conf['data']['recompute_normalization'] = False
#         normalizer.train(verbose=True)
#         # KGF: may want to set it back...
#         # normalizer.conf['data']['recompute_normalization'] = conf['data']['recompute_normalization']   # noqa
#         loader = Loader(conf, normalizer)
#         g.print_unique("...done")

#         # TODO(KGF): both preprocess.py and normalize.py are littered with print()
#         # calls that should probably be replaced with print_unique() when they are not
#         # purely loading previously-computed quantities from file
#         # (or we can continue to ensure that they are only ever executed by 1 rank)





def make_trainer(conf):
    if conf['framework'] == 'tensorflow' or conf['framework'] == 'tf':
        import tensorflow as tf

        # Prevent Keras TF backend deprecation messages from mpi_train() from
        # appearing jumbled with stdout, stderr msgs from above steps
        g.comm.Barrier()
        g.flush_all_inorder()

        if tf.__version__.startswith("2"):
            if conf['distributed']:
                from src.trainers.tensorflow2 import distributed_trainer
                trainer = distributed_trainer.distributed_trainer(conf)
            else:
                from src.trainers.tensorflow2 import trainer
                trainer = trainer.tf_trainer(conf)
        else:
            if conf['distributed']:
                from src.trainers.tensorflow1 import distributed_trainer
                trainer = distributed_trainer.distributed_trainer(conf)
            else:
                from src.trainers.tensorflow1 import trainer
                trainer = trainer.tf_trainer(conf)
    elif conf['framework'] == "torch":
        if conf['distributed']:
            from src.trainers.torch import distributed_trainer
            trainer = distributed_trainer.distributed_trainer(conf)
        else:
            from src.trainers.torch import trainer
            trainer = trainer.torch_trainer(conf)

    return trainer


def train(conf, trainer):
    trainer.initialize()
    trainer.batch_process()


def main():
    parser = make_parser()
    args = parser.parse_args()

    g.conf_file = args.conf_file
    if g.conf_file is not None:
        g.print_unique(f"Loading configuration from {g.conf_file}")
        conf = read_parameters(g.conf_file)
    elif os.path.exists('./conf.yaml'):
        g.print_unique("Loading configuration from ./conf.yaml")
        conf = read_parameters('./conf.yaml')
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                'conf.yaml')
    # add CLI switches to conf
    conf['distributed'] = args.distributed

    g.conf = conf

    # normalize(g.conf)
    trainer = make_trainer(g.conf)
    train(g.conf, trainer)


if __name__ == '__main__':
    main()
