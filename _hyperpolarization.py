__all__ = ["ChemShift","CoilSensitivities","Depolarisation","T2star",
           "fieldMap","MetaboliteConcentrationSignal",]

import sys
sys.path.append("../cmrsim")

from typing import Sequence, Union

import tensorflow as tf
import numpy as np

import cmrsim
from cmrsim.analytic.contrast.base import BaseSignalModel


class Depolarisation(BaseSignalModel):
    """ Model Based spectral-spatial excitation """
    #: From BaseSignalModel
    required_quantities = ("frequency_shift", )
    #: Nominal flip angle before frequency-shift adjustment; shape = (1, )
    flip_angle: tf.Variable
    #: Pulse bandwidth of the spatial-spectral excitation; shape = (1, )
    pulse_bandwidth: tf.Variable
    #: Echo index per repetition --> determines the expansion factor; shape = (None, )
    echo_index: tf.Variable

    def __init__(self, echo_index: tf.Tensor, flip_angle: tf.Tensor,
                 pulse_bandwidth: tf.Tensor, expand_repetitions: bool, **kwargs):
        """ Evaluate the flip angles for iso-chromates with a resonance frequency shifted with
        respect to the rotating frame.

        .. math::

            \\alpha_{v, m} = \\alpha_0 \left| sin \left(
              \\frac{f_m + b_0}{2 * BW_{SPSP}} \\right) \\right|^2

        :param echo_index: Indices of echos, determining the #repetition axis
        :param flip_angle: (1, ) in degree
        :param pulse_bandwidth: (1, ) in parts per million
        :param expand_repetitions: See BaseSignalModel for explanation
        """
        super().__init__(name="spsp_excitation", expand_repetitions=expand_repetitions, **kwargs)
        self.flip_angle = tf.Variable(flip_angle, shape=(1, ), dtype=tf.float32)
        self.pulse_bandwidth = tf.Variable(pulse_bandwidth, shape=(1,), dtype=tf.float32)
        self.echo_index = tf.Variable(echo_index, shape=(None, ), dtype=tf.int32)
        self.update()

    def update(self):
        super().update()
        self.expansion_factor = self.echo_index.read_value().shape[0]

    def __call__(self, signal_tensor: tf.Tensor, frequency_shift: tf.Tensor, **kwargs) -> tf.Tensor:
        """

        :param signal_tensor: (#particles, #repetitions, #k-space-samples) last two
                        dimensions can be 1
        :param frequency_shift: shift in parts per million per particle of shape
                        (#particles, #repetitions, #k-space-samples) last two dimensions can be 1,
        :return: - if expand_repetitions == True: (#voxel, #repetitions * #self.flip_angle,
                                            #k-space-samples)
                 - if expand_repetitions == False: (#voxel, #repetitions, #k-space-samples)
        """
        with tf.device(self.device):
            # All Cases : --> repetitions-axis of argument diffusion_tensor must be either 1 or
            # equal to self.expansion_factor
            tf.Assert(tf.shape(frequency_shift)[1] == 1 or
                      tf.shape(frequency_shift)[1] == self.expansion_factor,
                      ["Shape missmatch for input argument frequency_shift"])

            input_shape = tf.shape(signal_tensor)
            k_space_axis_tiling = tf.cast((tf.floor(input_shape[2] / tf.shape(frequency_shift)[2])),
                                          tf.int32)

            # Calculate actual flip-angle
            bw = tf.reshape(self.pulse_bandwidth, [1, 1, 1])
            fa = tf.reshape(self.flip_angle / 180. * np.pi, [1, 1, 1])
            sin_argument = frequency_shift / (2 * bw)
            actual_flip = fa * tf.abs(tf.sin(sin_argument)) ** 2  # (p, r_prev, ?)
            actual_flip = tf.tile(actual_flip, [1, 1, k_space_axis_tiling])  # (p, r_prev, k)

            echo_exponent = tf.cast(1 - self.echo_index, tf.float32)  # (exp_factor, )

            sin_alpha = tf.sin(actual_flip)  # (p, r_prev, k)
            cos_alpha = tf.cos(actual_flip)  # (p, r_prev, k)

            # Case 1: expand-dimensions
            if self.expand_repetitions or self.expansion_factor == 1:
                e_exp = tf.reshape(echo_exponent, [1, 1, -1, 1])
                depolarisation_factor = (tf.expand_dims(sin_alpha, 2) *
                                         tf.expand_dims(cos_alpha, 2) ** e_exp)

            # Case 2: repetition-axis of signal_tensor must match self.expansion_factor
            else:  # If run in tf.function this will likely cause a ResourceExhaustError
                tf.Assert(tf.shape(echo_exponent[0]) == tf.shape(actual_flip)[1],
                          ["Repetitions axis of frequency_shift does not match the expansion factor",
                           tf.shape(echo_exponent[0]), tf.shape(actual_flip)[1]])
                e_exp = tf.reshape(echo_exponent, [1, -1, 1])
                depolarisation_factor = sin_alpha * cos_alpha ** e_exp

            depolarisation_factor = tf.complex(depolarisation_factor,
                                               tf.zeros_like(depolarisation_factor))
            return signal_tensor * depolarisation_factor


class MetaboliteConcentrationSignal(BaseSignalModel):
    """ Evaluation of the concentration curve model leading to available signal"""
    #: from BasesignalModel
    required_quantities = ("metabolite_type", )
    #: Timings of acquisition (#reps, #samples). Set #samples == 1 to only use the evaluation at
    # echo time for all samples
    dynamic_scan_times: tf.Variable
    #: Conversion rates [$k_{PL}, k_{LP}, k_{PB}, k_{PA}$] for the metabolites
    # (Pyruvate, Lactate, bic???, ala???)
    conversion_rates = tf.Variable

    def __init__(self, dynamic_scan_times: tf.Tensor,
                 conversion_rates: Sequence[Union[float, tf.Tensor]],
                 myo_blood_flow: tf.Tensor, expand_repetitions: bool):
        """ For a given metabolite type :math:`m` of given particle :math:`p`, evaluates the available
        transverse magnetization at defined dynamic_scan times :math:`t_d` according to the equation:

        .. math::

            \\rho_{p, m}(t_d) = C_{p, m} e^{- \\frac{t_d}{(T_1)_{p, m}}},

        where the concentration is calculated by evaluating the solution to:

        .. math::

            \\frac{d}{dt_d} \begin{bmatrix}
                C_{pyr}(t_d) \\ C_{pyr}(t_d) \\ C_{pyr}(t_d) \\ C_{pyr}(t_d)
                 \end{bmatrix} = \begin{bmatrix}
                    (-k_{PL}-k_{PB}-k_{PA}) & k_{LP} & 0 & 0\\
                    k_{PL} & -k_{LP} & 0 & 0\\
                    k_{PB} & 0 & 0 & 0\\
                    k_{PA} & 0 & 0 & 0\\
                \end{bmatrix} \begin{bmatrix}
                 C_{pyr}(t_d) \\ C_{pyr}(t_d) \\ C_{pyr}(t_d) \\ C_{pyr}(t_d)
                 \end{bmatrix} + \begin{bmatrix}
                  MBF \cdot IRF(t_d) \cdot C_{LV}(td) \\ 0 \\ 0 \\ 0
                 \end{bmatrix}   \\ with: \\
                 MBF = Myocardial blood flow \\
                 IRF = Fermi-shaped impulse residue function \\
                 C_{LV} = LV blood pool signal of pyruvate

        :param dynamic_scan_times: in ms
        :param conversion_rates: in 1/ms
        :param myo_blood_flow:
        :param expand_repetitions:
        """
        super().__init__(name="metabolite_concentration", expand_repetitions=expand_repetitions)
        self.dynamic_scan_times = tf.Variable(dynamic_scan_times, dtype=tf.float32,
                                              shape=(None, None))
        self.conversion_rates = tf.Variable(conversion_rates, dtype=tf.float32, shape=(4, ))
        self.update()

    def update(self):
        super().update()
        self.expansion_factor = self.dynamic_scan_times.read_value().shape[0]

    def _eval_dynamic_model(self) -> tf.Tensor:
        """

        :return: (4, *self.dynamic_scan_times.shape)
        """
        reps, kspace = tf.unstack(tf.shape(self.dynamic_scan_times))
        return tf.ones([4, reps, kspace])

    def __call__(self, signal_tensor: tf.Tensor, metabolite_type: tf.Tensor,
                 T1: tf.Tensor, **kwargs):
        """

        :param signal_tensor:
        :param metabolite_type: integers between 0-3 (#particles, )
        :param T1: relaxation time in milliseconds (#particles, 1, 1)
        :return:
        """
        tf.Assert(tf.shape(T1)[1] == 1 or tf.shape(T1)[1] == self.expansion_factor,
                  ["Shape missmatch for input argument T1"])
        metabolite_concentrations = self._eval_dynamic_model()  # (4, reps, kspace or 1)
        decay_factors = tf.exp(- self.dynamic_scan_times[tf.newaxis] / T1)  # (p, reps, kspace or 1)
        c_per_particle = tf.gather(metabolite_concentrations, metabolite_type, axis=0)
        return c_per_particle * decay_factors

