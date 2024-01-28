cimport cython

import cython.parallel

cimport numpy as np

import numpy as np


#this function is no longer used for anything except a comparison test of the beamform/tied_array function
@cython.boundscheck(False)
@cython.wraparound(False)
def tiedbeam_apply_phase_sum(
        np.ndarray[np.complex64_t, ndim=2] baseband,
        np.ndarray[np.complex64_t, ndim=1] phases,
        np.ndarray[np.int32_t, ndim=2] masks,
        ):


    cdef int ninput = baseband.shape[0]
    cdef int ntime = baseband.shape[1]
    cdef int nmasks = masks.shape[0]

    cdef np.ndarray[np.complex64_t, ndim=2] out = np.zeros((nmasks, ntime),
                                                           dtype=np.complex64)

    cdef np.complex64_t tmp

    cdef int iit, jjm, kki
    for iit in cython.parallel.prange(ntime, nogil=True):
        for jjm in range(nmasks):
            for kki in range(ninput):
                tmp = baseband[kki, iit]
                tmp = tmp * phases[kki]
                tmp = tmp * masks[jjm, kki]
                out[jjm, iit] = out[jjm, iit] + tmp
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def unpack_baseband(baseband_array):

    baseband_array = np.ascontiguousarray(baseband_array)
    cdef np.ndarray[np.uint8_t, ndim=1] dflat = baseband_array.flat[:]
    cdef np.ndarray[np.complex64_t, ndim=1] outflat = np.empty(dflat.view().shape,
                                                               dtype=np.complex64)
    cdef int ii
    cdef np.complex64_t cnan
    cnan.real = float("NAN")
    cnan.imag = float("NAN")

    for ii in cython.parallel.prange(dflat.shape[0], nogil=True):
    #for ii in range(dflat.shape[0]):
        if dflat[ii] == 0:
            outflat[ii] = cnan
        else:
            outflat[ii].real = ((dflat[ii] >> 4) & 0x0F) - 8
            outflat[ii].imag = ((dflat[ii] & 0x0F) - 8)

    out = outflat.view()
    out.shape = baseband_array.view().shape
    return out


def unpack_baseband_transpose(np.ndarray[np.uint8_t, ndim=2] baseband_array):

    cdef int n0 = baseband_array.shape[0]
    cdef int n1 = baseband_array.shape[1]
    cdef np.ndarray[np.complex64_t, ndim=2] out = np.empty((n1, n0),
                                                           dtype=np.complex64)
    cdef int iib, ii, jj
    cdef int bsize = 64    # Cache line.
    cdef int nb = n1 // bsize
    if n1 % bsize:
        raise ValueError("Size of dimension 0 (%d) must divide evenly into %d"
                         % (n1, bsize))

    cdef np.complex64_t cnan, tout
    cdef np.uint8_t tin
    cnan.real = float("NAN")
    cnan.imag = float("NAN")

    for iib in cython.parallel.prange(0, n1, bsize, nogil=True):
        for jj in range(n0):
            for ii in range(bsize):
                tin = baseband_array[jj, iib + ii]
                if tin == 0:
                    tout = cnan
                else:
                    tout.real = ((tin >> 4) & 0x0F) - 8
                    tout.imag = (tin & 0x0F) - 8
                out[iib + ii, jj] = tout
    return out