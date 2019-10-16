from libc cimport math
import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cdef:
    # long IDX_CONVERSION_BIAS = 0
    # long IDX_CONVERSION_OFFSET = 0
    # long IDX_DELAY_BIAS = int(5 + 1)# int(math.pow(2, 24) + 1)
    # long IDX_DELAY_OFFSET = int(5) # int(math.pow(2, 24) + 2)
    # long len_weights = int((5) * 2) # (<long>(math.pow(2, 24)) + 1) * 2
    long IDX_CONVERSION_BIAS = 0
    long IDX_CONVERSION_OFFSET = 1
    long IDX_DELAY_BIAS = int(math.pow(2, 24) + 1)
    long IDX_DELAY_OFFSET = int(math.pow(2, 24) + 2)
    long len_weights = <long>((math.pow(2, 24) + 1) * 2)


def calc_gradient(
        np.uint32_t[:] feature_data,
        int[:] feature_positions,
        np.ndarray[np.npy_bool, ndim=1, cast=True] labels,
        DTYPE_t[:, :] scores,
        np.ndarray[np.int_t, ndim=1] timestamps
):
    cdef:
        int data_len = feature_positions.shape[0] - 1
        np.ndarray[DTYPE_t, ndim=1] gradients = np.zeros(len_weights, dtype=np.float64)
        int label
        DTYPE_t p
        DTYPE_t l
        DTYPE_t t
        DTYPE_t coeff_c, coeff_d
        int index
        int j
        np.uint32_t d
        np.uint32_t[:] feature

    for index in range(data_len):
        feature = feature_data[feature_positions[index]:feature_positions[index+1]]
        label = labels[index]
        p = scores[index][0]
        l = scores[index][1]
        t = timestamps[index]

        coeff_c = p * (1 - p)
        coeff_d = l

        if label:
            coeff_c *= -1.0 / p
            coeff_d *= t - 1.0 / l
        else:
            coeff_c *= (1 - math.exp(-l * t)) / (1 - p + p * math.exp(-l * t))
            coeff_d *= (p * t * math.exp(-l * t)) / (1 - p + p * math.exp(-l * t))

        gradients[IDX_CONVERSION_BIAS] += coeff_c
        gradients[IDX_DELAY_BIAS] += coeff_d

        for j in range(feature.shape[0]):
            d = feature[j]
            gradients[IDX_CONVERSION_OFFSET + d] += coeff_c
            gradients[IDX_DELAY_OFFSET + d] += coeff_d
    return gradients


def calc_scores(
        np.uint32_t[:] feature_data,
        int[:] feature_positions,
        np.ndarray[DTYPE_t] weights,
):
    cdef:
        int data_len = feature_positions.shape[0] - 1
        np.ndarray[DTYPE_t, ndim=2] scores = np.ndarray((data_len, 2), dtype=np.float64)
        DTYPE_t b_conversion, b_delay # value of basis function
        DTYPE_t p # value of sigmoid
        DTYPE_t lmd # value of exp(dot(w_delay, x))
        int index
        np.uint32_t[:] feature
        DTYPE_t sum_conversion = 0.0
        DTYPE_t sum_delay = 0.0

    for index in range(data_len):
        feature = feature_data[feature_positions[index]:feature_positions[index+1]]

        sum_conversion = 0.0
        sum_delay = 0.0
        for i in range(feature.shape[0]):
            sum_conversion += weights[IDX_CONVERSION_OFFSET + feature[i]]
            sum_delay += weights[IDX_DELAY_OFFSET + feature[i]]

        b_conversion = weights[IDX_CONVERSION_BIAS] + sum_conversion
        b_delay = weights[IDX_DELAY_BIAS] + sum_delay

        p = 1.0 / (1 + math.exp(-b_conversion))
        lmd = math.exp(b_delay)
        scores[index][0] = p
        scores[index][1] = lmd
    return scores
