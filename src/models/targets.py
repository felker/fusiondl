import numpy as np
from abc import ABC, abstractmethod

from src.utils.evaluation import (
    mse_np, binary_crossentropy_np, hinge_np,
    )
import tensorflow as tf  # noqa
from tensorflow.keras.losses import hinge

# remapper() method, used only in normalize.py, implicitly knows the
# transformation applied to Shot.ttd within Shot.convert_to_ttd()


# Requirement: larger value must mean disruption more likely.
class Target(ABC):
    activation = 'linear'
    loss = 'mse'

    @abstractmethod
    def loss_np(y_true, y_pred):
        from plasma.conf import conf
        return conf['model']['loss_scale_factor']*mse_np(y_true, y_pred)

    @abstractmethod
    def remapper(ttd, T_warning):
        # TODO(KGF): base class directly uses ttd=log(TTD+dt/10) quantity
        return -ttd


class BinaryTarget(Target):
    activation = 'sigmoid'
    loss = 'binary_crossentropy'

    @staticmethod
    def loss_np(y_true, y_pred):
        from plasma.conf import conf
        return (conf['model']['loss_scale_factor']
                * binary_crossentropy_np(y_true, y_pred))

    @staticmethod
    def remapper(ttd, T_warning, as_array_of_shots=True):
        # TODO(KGF): see below comment in HingeTarget.remapper()
        binary_ttd = 0*ttd
        mask = ttd < np.log10(T_warning)
        binary_ttd[mask] = 1.0
        binary_ttd[~mask] = 0.0
        return binary_ttd


class LogTTDTarget(Target):
    activation = 'linear'
    loss = 'mse'

    @staticmethod
    def loss_np(y_true, y_pred):
        from plasma.conf import conf
        return conf['model']['loss_scale_factor']*mse_np(y_true, y_pred)

    @staticmethod
    def remapper(ttd, T_warning):
        mask = ttd < np.log10(T_warning)
        ttd[~mask] = np.log10(T_warning)
        return -ttd


class TTDInvTarget(Target):
    activation = 'linear'
    loss = 'mse'

    @staticmethod
    def loss_np(y_true, y_pred):
        return mse_np(y_true, y_pred)

    @staticmethod
    def remapper(ttd, T_warning):
        eps = 1e-4  # hardcoded "avoid division by zero"
        ttd = 10**(ttd)  # see below comment about undoing log transformation
        mask = ttd < T_warning
        ttd[~mask] = T_warning
        ttd = (1.0)/(ttd + eps)
        return ttd


class TTDLinearTarget(Target):
    activation = 'linear'
    loss = 'mse'

    @staticmethod
    def loss_np(y_true, y_pred):
        from plasma.conf import conf
        return conf['model']['loss_scale_factor']*mse_np(y_true, y_pred)

    @staticmethod
    def remapper(ttd, T_warning):
        # TODO(KGF): this next line "undoes" the log-transform in shots.py
        # Shot.convert_to_ttd() (except for small offset of +dt/10)
        ttd = 10**(ttd)
        mask = ttd < T_warning
        ttd[~mask] = 0  # T_warning
        ttd[mask] = T_warning - ttd[mask]  # T_warning
        return ttd


# implements a "maximum" driven loss function. Only the maximal value in the
# time sequence is punished. Also implements class weighting
class MaxHingeTarget(Target):
    activation = 'linear'
    loss = 'hinge'
    fac = 1.0

    @staticmethod
    def loss(y_true, y_pred):
        # TODO(KGF): this function is unused and unique to this class
        from plasma.conf import conf
        fac = MaxHingeTarget.fac
        # overall_fac =
        # np.prod(np.array(K.shape(y_pred)[1:]).astype(np.float32))
        overall_fac = K.prod(K.cast(K.shape(y_pred)[1:], K.floatx()))
        max_val = K.max(y_pred, axis=-2)  # temporal axis!
        max_val1 = K.repeat(max_val, K.shape(y_pred)[-2])
        mask = K.cast(K.equal(max_val1, y_pred), K.floatx())
        y_pred1 = mask * y_pred + (1-mask) * y_true
        weight_mask = K.mean(y_true, axis=-1)
        weight_mask = K.cast(K.greater(weight_mask, 0.0),
                             K.floatx())  # positive label!
        weight_mask = fac*weight_mask + (1 - weight_mask)
        # return weight_mask*squared_hinge(y_true, y_pred1)

        # KGF: this is the only place where tensorflow.keras.losses.hinge()
        # is used in this file
        return conf['model']['loss_scale_factor'] * \
            overall_fac*weight_mask*hinge(y_true, y_pred1)

    @staticmethod
    def loss_np(y_true, y_pred):
        from plasma.conf import conf
        # TODO(KGF): fac = positive_example_penalty is only used in this class,
        # only in above (unused) loss() fn, which only this class has, and is
        # never called. 2 lines related to fac commented-out in this fn.
        #
        # fac = MaxHingeTarget.fac
        overall_fac = np.prod(np.array(y_pred.shape).astype(np.float32))
        max_val = np.max(y_pred, axis=-2)  # temporal axis!
        max_val = np.reshape(
            max_val, max_val.shape[:-1] + (1,) + (max_val.shape[-1],))
        max_val = np.tile(max_val, (1, y_pred.shape[-2], 1))
        mask = np.equal(max_val, y_pred)
        mask = mask.astype(np.float32)
        y_pred = mask * y_pred + (1-mask) * y_true
        # positive label! weight_mask = fac*weight_mask + (1 - weight_mask):
        weight_mask = np.greater(y_true, 0.0).astype(np.float32)
        # return np.mean(
        #  weight_mask*np.square(np.maximum(1. - y_true * y_pred, 0.)))
        # , axis=-1)
        # only during training, here we want to completely sum up over all
        # instances
        return (conf['model']['loss_scale_factor']
                * np.mean(overall_fac * weight_mask
                          * np.maximum(1. - y_true * y_pred, 0.)))

    @staticmethod
    def remapper(ttd, T_warning, as_array_of_shots=True):
        # TODO(KGF): see below comment in HingeTarget.remapper()
        binary_ttd = 0*ttd
        mask = ttd < np.log10(T_warning)
        binary_ttd[mask] = 1.0
        binary_ttd[~mask] = -1.0
        return binary_ttd


class HingeTarget(Target):
    activation = 'linear'
    loss = 'hinge'

    @staticmethod
    def loss_np(y_true, y_pred):
        from plasma.conf import conf
        return conf['model']['loss_scale_factor']*hinge_np(y_true, y_pred)
        # return squared_hinge_np(y_true, y_pred)

    @staticmethod
    def remapper(ttd, T_warning, as_array_of_shots=True):
        # TODO(KGF): zeros the ttd=log(TTD+dt/10) (just reuses shape)
        binary_ttd = 0*ttd
        mask = ttd < np.log10(T_warning)
        binary_ttd[mask] = 1.0
        binary_ttd[~mask] = -1.0
        return binary_ttd
