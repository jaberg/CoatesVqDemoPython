import os
import time

import numpy as np
import scipy.io

from skdata.utils.glviewer import glumpy_viewer, command, glumpy

#import pyll
from .cifar10 import CF10

import theano
import theano.tensor as tensor
from theano.tensor.nnet import conv

from .utils import mean_and_std
from .utils import assert_allclose


def boxconv((x, x_shp), kershp, channels=False):
    """
    channels: sum over channels (T/F)
    """
    kershp = tuple(kershp)
    if channels:
        rshp = (   x_shp[0],
                    1,
                    x_shp[2] - kershp[0] + 1,
                    x_shp[3] - kershp[1] + 1)
        kerns = np.ones((1, x_shp[1]) + kershp, dtype=x.dtype)
    else:
        rshp = (   x_shp[0],
                    x_shp[1],
                    x_shp[2] - kershp[0] + 1,
                    x_shp[3] - kershp[1] + 1)
        kerns = np.ones((1, 1) + kershp, dtype=x.dtype)
        x_shp = (x_shp[0]*x_shp[1], 1, x_shp[2], x_shp[3])
        x = x.reshape(x_shp)
    try:
        rval = tensor.reshape(
                conv.conv2d(x,
                    theano.shared(kerns),
                    image_shape=x_shp,
                    filter_shape=kerns.shape,
                    border_mode='valid'),
                rshp)
    except Exception, e:
        if "Bad size for the output shape" in str(e):
            raise InvalidDescription('boxconv', (x_shp, kershp, channels))
        else:
            raise
    return rval, rshp

#@pyll.scope.define
def random_patches(images, N, R, C):
    """Return a stack of N image patches"""
    n_imgs, iR, iC, iF = images.shape
    rval = np.empty((N, R, C, iF), dtype=images.dtype)
    rng = np.random.RandomState(1234)
    for rv_i in rval:
        src = rng.randint(n_imgs)
        roffset = rng.randint(iR - R)
        coffset = rng.randint(iC - C)
        rv_i[:] = images[src, roffset: roffset + R, coffset: coffset + C]
    return rval


def coates_patches(images, N, R, C):
    assert len(images) == 50000
    assert N == 50000
    assert R == C == 6
    patches = []
    m_patches = scipy.io.loadmat(mpath('patches.mat'))['patches']
    m_patches_rci = scipy.io.loadmat(mpath('patches_rci.mat'))['patches_rci']
    #print 'm_patches',
    #print m_patches.shape, m_patches.min(), m_patches.max()
    for j, (r, c, i) in enumerate(m_patches_rci):
        patch = images[i - 1, c - 1:c + 5, r - 1:r + 5]
        flatpatch = patch.transpose(2, 0, 1).flatten()
        patches.append(flatpatch)
        assert np.allclose(flatpatch, m_patches[j])
    patches = np.asarray(patches)
    assert np.allclose(patches, m_patches)
    return patches


def coates_dictionary(X):
    foo = scipy.io.loadmat(mpath('dictionary.mat'))
    #print foo['dictionary_elems'][:, 0]
    dictionary = X[foo['dictionary_elems'].flatten() - 1]
    dictionary = dictionary / (np.sqrt((dictionary ** 2).sum(axis=1))[:, None] + 1e-20);
    # tolerances are getting a little wacky, but the images look right
    assert_allclose(dictionary, foo['dictionary'], atol=1e-6, rtol=0.03)
    return dictionary


def contrast_normalize(patches):
    X = patches
    N = X.shape[1]
    unbias = float(N) / (float(N) - 1)
    xm = X.mean(1)
    xv = unbias * X.var(1)
    X = (X - xm[:, None]) / np.sqrt(xv[:, None] + 10)
    return X


def summarize(msg, X):
    print msg, X.shape, X.min(), X.max(), X.mean()


#@pyll.scope.define_info(o_len=2)
def patch_whitening_filterbank(patches, retX=False, reshape=True):
    """
    Image patches of uint8 pixels

    """
    # Algorithm from Coates' sc_vq_demo.m
    assert str(patches.dtype) == 'uint8'

    # -- patches -> column vectors
    X = patches.reshape(len(patches), -1).astype('float64')

    X = contrast_normalize(X)

    # -- ZCA whitening (with low-pass)
    M, _std = mean_and_std(X)
    #M = X.mean(0)  -- less numerically accurate?
    Xm = X - M
    assert Xm.shape == X.shape
    C = np.dot(Xm.T, Xm) / (Xm.shape[0] - 1)
    D, V = np.linalg.eigh(C)
    P = np.dot(np.sqrt(1.0 / (D + 0.1)) * V, V.T)

    # -- return to image space
    if reshape:
        M = M.reshape(patches.shape[1:])
        P = P.reshape((P.shape[0],) + patches.shape[1:])

    if retX:
        return M, P, X
    else:
        return M, P


#@pyll.scope.define_info(o_len=2)
def cifar10_img_classification_task(dtype='float32'):
    imgs, labels = CF10.img_classification_task(dtype='float32')
    return imgs, labels


def im2col(img, (R, C)):
    H, W, F = img.shape
    rval = np.zeros(((H - R + 1), (W - C + 1), R, C, F), dtype=img.dtype)
    for ii in xrange(rval.shape[0]):
        for jj in xrange(rval.shape[1]):
            rval[ii, jj] = img[ii:ii + R, jj: jj+ C]
    return rval


def show_centroids(D):
    D = D.copy()
    for di in D:
        di -= di.min()
        di /= di.max()
    glumpy_viewer(
            img_array=D.astype('float32'),
            arrays_to_print=[],
            )


import line_profiler
profile = line_profiler.LineProfiler()
import time


#@profile
def extract_features(imgs, D, M, P, alpha, R, C,
        internal_dtype='float64'):
    tt = time.time()
    N, H, W, F = imgs.shape
    numBases = len(D)
    M = M.flatten().astype(internal_dtype)
    P = P.reshape((R * C * 3, R * C * 3)).astype(internal_dtype)
    XC = np.zeros((N, len(D), 2, 2, 2), dtype='float32')
    PD = np.dot(P, D.reshape(len(D), -1).T).astype(internal_dtype)
    for i in xrange(len(imgs)):
        print 'PY ITER', i
        tt = time.time()
        if 0 == i % 100:
            #profile.print_stats()
            print i, (time.time() - tt)
        patches = im2col(imgs[i], (R, C)).astype(internal_dtype)
        patches = patches.transpose(0, 1, 4, 2, 3).reshape(
                (patches.shape[0] * patches.shape[1], -1))

        patches_cn = contrast_normalize(patches)
        z = np.dot(patches_cn - M, PD)

        summarize('z', z)

        if 0:
            aPDx = p_scale[:, None] * np.dot(patches, PD)
            aPDmmm = (p_scale * p_mean)[:, None] * PD.sum(0)

            #tmp_1 = np.dot((patches - p_mean[:, None]) * p_scale[:, None], PD)
            #tmp_2 = aPDx - aPDmmm
            #print 'tmp1', tmp_1
            #print 'tmp2', tmp_2
            #assert np.allclose(tmp_1, tmp_2)

            PDM = np.dot(M, PD)

            Z = aPDx - aPDmmm - PDM
            for foo in [
                    #aPDx,
                    #-aPDmmm,
                    #-PDM,
                    #aPDx - aPDmmm,
                    #-aPDmmm - PDM,
                    #z,
                    #Z,
                    ]:
                print '->', foo.min(), foo.max(), foo.mean(), foo.shape
                #print foo.flatten()[:0]
                #print foo
                pass
            #print z
            #print Z
            print 'PY ATOL', abs(z - Z).max()
            print 'PY RTOL', (abs(z - Z) / ((1e-12 + abs(z) + abs(Z)))).max()

        prows = H - R + 1
        pcols = W - C + 1
        z = z.reshape((prows, pcols, numBases))
        hr = int(np.round(prows / 2.))
        hc = int(np.round(pcols / 2.))
        XC[i, :, 0, 0, 0] = np.maximum(z[:hr, :hc] - alpha, 0).sum(1).sum(0)
        XC[i, :, 0, 0, 1] = -np.maximum(-z[:hr, :hc] - alpha, 0).sum(1).sum(0)
        XC[i, :, 0, 1, 0] = np.maximum(z[:hr, hc:] - alpha, 0).sum(1).sum(0)
        XC[i, :, 0, 1, 1] = -np.maximum(-z[:hr, hc:] - alpha, 0).sum(1).sum(0)
        XC[i, :, 1, 0, 0] = np.maximum(z[hr:, :hc] - alpha, 0).sum(1).sum(0)
        XC[i, :, 1, 0, 1] = -np.maximum(-z[hr:, :hc] - alpha, 0).sum(1).sum(0)
        XC[i, :, 1, 1, 0] = np.maximum(z[hr:, hc:] - alpha, 0).sum(1).sum(0)
        XC[i, :, 1, 1, 1] = -np.maximum(-z[hr:, hc:] - alpha, 0).sum(1).sum(0)
    return XC


def extract_features_theano(imgs, D, M, P, alpha, R, C,
        internal_dtype='float64', batchsize=1):
    tt = time.time()
    N, H, W, F = imgs.shape
    numBases = len(D)
    M = M.flatten().astype(internal_dtype)
    P = P.reshape((R * C * 3, R * C * 3)).astype(internal_dtype)
    XC = np.zeros((N, len(D), 2, 2, 2), dtype='float32')
    PD = np.dot(P, D.reshape(len(D), -1).T).astype(internal_dtype)

    s_imgs = theano.shared(imgs[:batchsize].astype(internal_dtype))
    sXC_base = sXC = theano.shared(XC[:2])
    x_shp = (batchsize, F, H, W)
    ker_shape = (R, C)

    s_foo = []

    # -- calculate patch means, patch variances
    p_sum, _shp = boxconv((s_imgs, x_shp), ker_shape, channels=True)
    p_ssq, _shp = boxconv((s_imgs ** 2, x_shp), ker_shape, channels=True)
    p_mean = p_sum / (R * C * F)
    p_var = p_ssq / (R * C * F) - (p_sum / (R * C * F)) ** 2
    unbias = float(R * C * F) / (R * C * F - 1)
    #s_foo.append(p_mean)
    #s_foo.append(unbias * p_var)
    p_scale = 1.0 / tensor.sqrt(unbias * p_var + 10)
    assert p_mean.ndim == 4
    assert p_scale.ndim == 4
    assert p_mean.broadcastable[1]
    assert p_scale.broadcastable[1]

    # --
    # from whitening, we have a shift and linear transform (P)
    # for each patch (as vector).
    #
    # let m be the vector [m m m m] that replicates p_mean
    # let a be the scalar p_scale
    # let x be an image patch from s_imgs
    #
    # Whitening means applying the affine transformation
    #   (c - M) P
    # to contrast-normalized patch c = a (x - m),
    # where a = p_scale and m = p_mean.
    #
    # We also want to extract features in dictionary D
    #
    #   (c - M) P D
    #   = (a (x - [m,m,m]) - M) P D
    #   = (a x - a [m,m,m] - M) P D
    #   = a x P D - a [m,m,m] P D - M P D
    #

    PD_kerns = PD.reshape(3, 6, 6, numBases)\
            .transpose(3, 0, 1, 2)[:, :, ::-1, ::-1]
    s_PD_kerns = theano.shared(np.asarray(PD_kerns, order='C'))

    PDx = conv.conv2d(
            s_imgs - 128,
            s_PD_kerns,
            image_shape=x_shp,
            filter_shape=(numBases, 3, 6, 6),
            border_mode='valid')

    s_PD_sum = theano.shared(PD.sum(0))
    PDmmm = (p_mean - 128) * s_PD_sum.dimshuffle(0, 'x', 'x')

    s_PDM = theano.shared(np.dot(M, PD))  # -- vector

    z = p_scale * (PDx - PDmmm) - s_PDM.dimshuffle(0, 'x', 'x')
    assert z.ndim == 4
    #s_foo.append(z)

    hr = int(np.round((H - R + 1) / 2.))
    hc = int(np.round((W - C + 1) / 2.))

    sXC = tensor.set_subtensor(sXC[:, :, 0, 0, 0],
        tensor.maximum(z[:, :, :hr, :hc] - alpha, 0).sum([2, 3]))
    sXC = tensor.set_subtensor(sXC[:, :, 0, 0, 1],
        -tensor.maximum(-z[:, :, :hr, :hc] - alpha, 0).sum([2, 3]))
    sXC = tensor.set_subtensor(sXC[:, :, 0, 1, 0],
        tensor.maximum(z[:, :, :hr, hc:] - alpha, 0).sum([2, 3]))
    sXC = tensor.set_subtensor(sXC[:, :, 0, 1, 1],
        -tensor.maximum(-z[:, :, :hr, hc:] - alpha, 0).sum([2, 3]))
    sXC = tensor.set_subtensor(sXC[:, :, 1, 0, 0],
        tensor.maximum(z[:, :, hr:, :hc] - alpha, 0).sum([2, 3]))
    sXC = tensor.set_subtensor(sXC[:, :, 1, 0, 1],
        -tensor.maximum(-z[:, :, hr:, :hc] - alpha, 0).sum([2, 3]))
    sXC = tensor.set_subtensor(sXC[:, :, 1, 1, 0],
        tensor.maximum(z[:, :, hr:, hc:] - alpha, 0).sum([2, 3]))
    sXC = tensor.set_subtensor(sXC[:, :, 1, 1, 1],
        -tensor.maximum(-z[:, :, hr:, hc:] - alpha, 0).sum([2, 3]))

    z_fn = theano.function([], s_foo,
            updates={sXC_base: sXC})

    i = 0
    sXC_base.set_value(XC[i:i + batchsize])
    while i < len(imgs):
        print 'THEANO ITER', i
        tt = time.time()
        s_imgs.set_value(
                imgs[i:i + batchsize].transpose(0, 3, 1, 2).astype('float32'))
        vfoo = z_fn()
        for foo in vfoo:
            print ' -> ', foo.min(), foo.max(), foo.mean(), foo.shape
            #print foo
        XC[i:i + batchsize] = sXC_base.get_value()
        i += batchsize
        print 'TIME', time.time() - tt

    return XC


def mpath(name):
    return os.path.join(
        '/home/bergstra/.VENV/eccv12/src/sc_vq_demo/',
        name)


def track_matlab():
    """
    Mar 28 - this function gets exactly the same features as Adam Coates'
    matlab code.
    """

    imgs, labels = CF10.img_classification_task(dtype='uint8')
    patches = coates_patches(imgs[:50000], 50000, 6, 6)
    M, P, patches_cn = patch_whitening_filterbank(patches, retX=True, reshape=False)

    patches_w = np.dot(patches_cn - M, P)

    if 1: # -- verify M, P and patches_w
        foo = scipy.io.loadmat(mpath('patches_final.mat'))
        assert_allclose(M, foo['M'].flatten())
        assert_allclose(P, foo['P'])
        assert_allclose(patches_w[0], foo['patches'][0], atol=1e-6, rtol=1e-3)
        assert_allclose(patches_w, foo['patches'], atol=1e-6, rtol=1e-3)
        del foo

    dictionary = coates_dictionary(patches_w)

    if 0:
        show_centroids(
                dictionary.reshape(len(dictionary), 3, 6, 6).transpose(0, 3, 2,
                    1))

    m_trainXC = scipy.io.loadmat(mpath('trainXC5.mat'))['trainXC']

    nF = len(dictionary)
     # -- check that float64 feature extraction works
    trainXC = extract_features(imgs[:5], dictionary, M, P, .25, 6, 6,
            internal_dtype='float64')
    trainXCp = trainXC.transpose(0, 2, 3, 4, 1).reshape(5, -1)
    assert_allclose(trainXCp, m_trainXC, atol=1e-4, rtol=1e-4)

    if 0:
        # -- check that float32 feature extraction is good enough
        trainXC = extract_features(imgs[:5], dictionary, M, P, .25, 6, 6,
                internal_dtype='float32')
        trainXCp = trainXC.transpose(0, 2, 3, 4, 1).reshape(5, -1)
        assert_allclose(trainXCp, m_trainXC, atol=1e-4, rtol=1e-4)

    if 0:
        # -- check that the theano version is correct
        trainXC = extract_features_theano(imgs[:5], dictionary, M, P, .25, 6, 6,
                internal_dtype='float64')
        trainXCp = trainXC.transpose(0, 2, 3, 4, 1).reshape(5, -1)
        assert_allclose(trainXCp, m_trainXC, atol=1e-4, rtol=1e-4)

    # -- check that the theano version is correct-ish
    trainXC = extract_features_theano(imgs[:5], dictionary, M, P, .25, 6, 6,
            internal_dtype='float32')
    trainXCp = trainXC.transpose(0, 2, 3, 4, 1).reshape(5, -1)
    assert_allclose(trainXCp, m_trainXC, atol=5e-3, rtol=5e-3)

    # -- compute the whole thing
    try:
        trainXC = np.load('train_XC.npy')
        testXC = np.load('test_XC.npy')
    except IOError:
        trainXC = extract_features_theano(imgs[:50000], dictionary, M, P, .25, 6, 6,
                internal_dtype='float32', batchsize=100)
        testXC = extract_features_theano(imgs[50000:], dictionary, M, P, .25, 6, 6,
                internal_dtype='float32', batchsize=100)
        np.save('train_XC.npy', trainXC)
        np.save('test_XC.npy', testXC)

    trainXC = trainXC.transpose(0, 2, 3, 4, 1).reshape(50000, -1)
    testXC = testXC.transpose(0, 2, 3, 4, 1).reshape(10000, -1)
    assert_allclose(trainXC[:5], m_trainXC, atol=5e-3, rtol=5e-3)
    dct = scipy.io.loadmat( mpath('trainXC_picked.mat'))
    for m_idx, m_fvec in zip(dct['picked_i'].flatten(), dct['trainXC_picked']):
        print 'spot check', m_idx
        assert_allclose(trainXC[m_idx - 1], m_fvec, atol=5e-3, rtol=5e-3)

    xmean, xstd = mean_and_std(trainXC, remove_std0=False, unbiased=True)
    xstd = np.sqrt(xstd ** 2 + 0.01) # -- hacky correction

    print 'loading stats'
    dct = scipy.io.loadmat(mpath('trainXCstats.mat'))

    m_xmean = dct['trainXC_mean']
    assert_allclose(xmean, trainXC.mean(0, dtype='float64'))
    assert_allclose(m_xmean, xmean)

    m_xstd = dct['trainXC_sd']
    assert_allclose(m_xstd, xstd)


def coates_classif():
    from pyll_slm import fit_linear_svm, model_predict, error_rate
    from .utils import linear_kernel, mean_and_std

    imgs, labels = CF10.img_classification_task(dtype='uint8')

    trainXC = np.load('train_XC.npy')
    testXC = np.load('test_XC.npy')

    trainXC = trainXC.reshape((len(trainXC), -1))
    testXC = testXC.reshape((len(testXC), -1))

    xmean, xstd = mean_and_std(trainXC, remove_std0=False, unbiased=True)
    xstd = np.sqrt(xstd ** 2 + 0.01) # -- hacky correction

    summarize('Xmean', xmean)
    summarize('Xstd', xstd)

    trainXC -= xmean
    trainXC /= xstd
    testXC -= xmean
    testXC /= xstd

    # XXX use e.g. liblinear or pyautodiff to solve this in primal using scipy
    #     l_bfgs_b.
    svm = fit_linear_svm(
            (trainXC, labels[:len(trainXC)]),
            solver=solver,
            l2_regularization=0.0025)

    pred = svm(trainXC)
    print 'TRAIN ERR', error_rate(pred, labels[:50000])

    pred = svm(testXC)
    print 'TEST ERR', error_rate(pred, labels[50000:])


