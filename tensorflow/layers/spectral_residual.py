import tensorflow as tf
from tensorflow.keras.layers import Layer, RepeatVector


class SpectralResidualBlock(Layer):
    def __init__(self, m=5, tau=1):
        super(SpectralResidualBlock, self).__init__()
        self.m = m
        self.k = m
        self.tau = tau

    def _input_estimated_sequence(self, x):
        assert self.m >= 2
        _gm = (x[:, :, -1:-2:-1] - x[:, :, -2:-(self.m+2):-1]) / tf.range(1, self.m+1, dtype=tf.float32)
        gm = tf.math.reduce_sum(_gm, axis=-1)
        _x_next = x[:, :, -self.m+1] + (gm*self.m)
        _x_next = RepeatVector(self.k)(_x_next)
        x_next = tf.transpose(_x_next, (0, 2, 1))

        return tf.concat([x, x_next], axis=-1)

    def call(self, x):
        x_added = self._input_estimated_sequence(x)

        x_added = tf.cast(x_added, dtype=tf.complex64)
        fft = tf.signal.fft(x_added)[:, :, :-self.m]
        A = tf.math.abs(fft)
        P = tf.math.angle(fft)
        L = tf.math.log(A)
        h_q = 1/(x.shape[-1]**2) * tf.ones((x.shape[-1], x.shape[-1]))
        AL = L @ h_q
        R = L - AL

        _P_imaginary = tf.cast(P, dtype=tf.complex64) * 1j
        _R_complex = tf.cast(P, dtype=tf.complex64)
        S = tf.math.abs(tf.signal.ifft(tf.math.exp(_R_complex + _P_imaginary)))
        
        _O = (S - tf.reduce_sum(S, axis=-1, keepdims=True)) / tf.reduce_sum(S, axis=-1, keepdims=True)
        _O = tf.abs(_O)
        O = tf.where(_O > self.tau, 1., 0.)

        return R, O
