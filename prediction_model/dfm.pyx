from libc cimport math
import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef:
    long IDX_BIAS = 0
    long IDX_OFFSET = 1

def calc_gradient_dfm(
        np.uint32_t[:] feature_data,
        int[:] feature_positions,
        long num_features,
        np.ndarray[np.npy_bool, ndim=1, cast=True] y,
        DTYPE_t[:, :] scores,
        long[:] timestamps
):
    cdef:
        long IDX_DELAY_BIAS = num_features + 1
        long IDX_DELAY_OFFSET = num_features + 2
        long LEN_WEIGHTS = (num_features + 1) * 2
        int data_len = feature_positions.shape[0] - 1
        np.ndarray[DTYPE_t] gradients = np.zeros(LEN_WEIGHTS, dtype=np.float64)
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
        label = y[index]
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

        gradients[IDX_BIAS] += coeff_c
        gradients[IDX_DELAY_BIAS] += coeff_d

        for j in range(feature.shape[0]):
            d = feature[j]
            gradients[IDX_OFFSET + d] += coeff_c
            gradients[IDX_DELAY_OFFSET + d] += coeff_d
    return gradients


def calc_scores_dfm(
        np.uint32_t[:] feature_data,
        int[:] feature_positions,
        long num_features,
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
        long IDX_DELAY_BIAS = num_features + 1
        long IDX_DELAY_OFFSET = num_features + 2
        long LEN_WEIGHTS = (num_features + 1) * 2

    for index in range(data_len):
        feature = feature_data[feature_positions[index]:feature_positions[index+1]]

        sum_conversion = 0.0
        sum_delay = 0.0
        for i in range(feature.shape[0]):
            sum_conversion += weights[IDX_OFFSET + feature[i]]
            sum_delay += weights[IDX_DELAY_OFFSET + feature[i]]

        b_conversion = weights[IDX_BIAS] + sum_conversion
        b_delay = weights[IDX_DELAY_BIAS] + sum_delay

        p = 1.0 / (1 + math.exp(-b_conversion))
        lmd = math.exp(b_delay)
        scores[index][0] = p
        scores[index][1] = lmd
    return scores

def calc_gradient_dl(
        long num_features,
        np.uint32_t[:] feature_data,
        int[:] feature_positions,
        np.ndarray[np.npy_bool, ndim=1, cast=True] y,
        np.ndarray[DTYPE_t] weights1,
        np.ndarray[DTYPE_t] weights2,
        ):
    cdef:
        int data_len = feature_positions.shape[0] - 1
        np.ndarray[DTYPE_t] gradients = np.zeros(num_features + 1, dtype=np.float64)
        int label
        DTYPE_t coeff,
        int index
        int j
        np.uint32_t[:] feature
        np.ndarray[DTYPE_t] f = predict(feature_data, feature_positions, weights1)
        np.ndarray[DTYPE_t] g = predict(feature_data, feature_positions, weights2)
        DTYPE_t f_i
        DTYPE_t g_i

    for index in range(data_len):
        feature = feature_data[feature_positions[index]:feature_positions[index+1]]
        label = y[index]
        f_i = f[index]
        g_i = g[index]

        diff_grad = - label * ( (1.0 / g_i) * (1 - f_i) - (1.0 - 1.0 / g_i) * f_i ) + (1 - label) * f_i
        gradients[IDX_BIAS] += diff_grad 
        for j in range(feature.shape[0]):
            gradients[IDX_OFFSET + feature[j]] += diff_grad
    return gradients

def predict(
        np.uint32_t[:] feature_data,
        int[:] feature_positions,
        np.ndarray[DTYPE_t] weights,
        ):

    cdef:
        int data_len = feature_positions.shape[0] - 1
        int index
        int i
        np.uint32_t[:] feature
        DTYPE_t pred = 0.0
        np.ndarray[DTYPE_t] preds = np.zeros(data_len, dtype=np.float64)

    for index in range(data_len):
        feature = feature_data[feature_positions[index]:feature_positions[index+1]]
        pred = 0 
        pred += weights[IDX_BIAS]
        for i in range(feature.shape[0]):
            pred += weights[IDX_OFFSET + feature[i]] 
        
        pred = 1.0 / (1.0 + math.exp(-pred))
        preds[index] = pred
            
    return preds

def gradients_lr(
        np.uint32_t[:] feature_data,
        int[:] feature_positions,
        long num_features,
        np.ndarray[np.npy_bool, ndim=1, cast=True] y,
        np.ndarray[DTYPE_t] weights,
        ):
    cdef:
        int data_len = feature_positions.shape[0] - 1
        np.ndarray[DTYPE_t] gradients = np.ndarray(num_features + 1, dtype=np.float64)
        int index
        np.uint32_t[:] feature
        DTYPE_t sig_Xw
        np.ndarray[DTYPE_t] f = predict(feature_data, feature_positions, weights)
        DTYPE_t diff_grad

    for index in range(data_len):
        sig_XW = 0.0
        feature = feature_data[feature_positions[index]:feature_positions[index+1]]

        
        diff_grad = f[index] - y[index]
        gradients[IDX_BIAS] += diff_grad
        for i in range(feature.shape[0]):
            gradients[IDX_OFFSET + feature[i]] += diff_grad
    return gradients 
