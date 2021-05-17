import itertools
import os
import numpy as np


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


def time_sensitive_interp(x, t, t_new):
    indices = np.maximum(0, np.searchsorted(t, t_new, side='right')-1)
    return x[indices]


def resample_signal(t, sig, tmin, tmax, dt, precision_str='float32'):
    order = np.argsort(t)
    t = t[order]
    sig = sig[order, :]
    sig_width = sig.shape[1]
    tt = np.arange(tmin, tmax, dt, dtype=precision_str)
    sig_interp = np.zeros((len(tt), sig_width), dtype=precision_str)
    for i in range(sig_width):
        # make sure to not use future information
        sig_interp[:, i] = time_sensitive_interp(sig[:, i], t, tt)
    if(np.any(np.isnan(sig_interp))):
        print("signal contains nan")
    if(np.any(t[1:] - t[:-1] <= 0)):
        print("non increasing")
        idx = np.where(t[1:] - t[:-1] <= 0)[0][0]
        print(t[idx-10:idx+10])

    return tt, sig_interp


def cut_signal(t, sig, tmin, tmax):
    mask = np.logical_and(t >= tmin,  t <= tmax)
    return t[mask], sig[mask, :]


def cut_and_resample_signal(t, sig, tmin, tmax, dt, precision_str):
    t, sig = cut_signal(t, sig, tmin, tmax)
    return resample_signal(t, sig, tmin, tmax, dt, precision_str)


def get_individual_shot_file(prepath, machine, shot_num, raw_signal=False,
                             ext='.txt'):
    """Return filepath of raw input .txt shot signal or processed .npz shot"""
    if raw_signal:
        return os.path.join(prepath, str(shot_num) + ext)
    else:
        return os.path.join(prepath, str(machine) + '_' + str(shot_num) + ext)


def append_to_filename(path, to_append):
    ending_idx = path.rfind('.')
    new_path = path[:ending_idx] + to_append + path[ending_idx:]
    return new_path


def train_test_split(x, frac, do_shuffle=False):
    # TODO(KGF): rename these 2x fns; used for generic ShotList.split_direct
    if not isinstance(x, np.ndarray):
        return train_test_split_robust(x, frac, do_shuffle)
    mask = np.array(range(len(x))) < frac*len(x)
    # Note, these functions do not directly split the "disruptive" subset of
    # ShotLists; they are only applied to the overall sets and rely on random
    # shuffling to produce the correct disruptive split in large N sample limit
    if do_shuffle:
        np.random.shuffle(mask)
    return x[mask], x[~mask]


def train_test_split_robust(x, frac, do_shuffle=False):
    mask = np.array(range(len(x))) < frac*len(x)
    if do_shuffle:
        np.random.shuffle(mask)
    train = []
    test = []
    for (i, _x) in enumerate(x):
        if mask[i]:
            train.append(_x)
        else:
            test.append(_x)
    return train, test


def concatenate_sublists(superlist):
    return list(itertools.chain.from_iterable(superlist))


def get_signal_slices(signals_superlist):
    indices_superlist = []
    signals_so_far = 0
    for sublist in signals_superlist:
        indices_sublist = signals_so_far + np.array(range(len(sublist)))
        signals_so_far += len(sublist)
        indices_superlist.append(indices_sublist)
    return indices_superlist
