import sys

import numpy as np
from cmrseq.core.bausteine import TrapezoidalGradient

sys.path.append("../cmrseq")
import math
from copy import deepcopy

import cmrseq
from pint import Quantity
import matplotlib.pyplot as plt


def hyperpol_sequence(fov: Quantity, matrix_size: Quantity, echo_times: Quantity,
                      spsp_bandwidth: Quantity, rr_intervals: Quantity,
                      echos_per_interval: int = 3,
                      adc_duration: Quantity = Quantity(0.5, "ms"),
                      pw: Quantity = Quantity(0.75, "ms"),
                      FA: Quantity = Quantity(30, "degree"),
                      slice_selective: bool = True,
                      pfl: int = 11):

    system_specs = cmrseq.SystemSpec(max_grad=Quantity(80, "mT/m"),
                                     max_slew=Quantity(200, "mT/m/ms"),
                                     grad_raster_time=Quantity(0.005, "ms"),
                                     rf_raster_time=Quantity(0.005, "ms"),
                                     adc_raster_time=Quantity(0.005, "ms"))
    if slice_selective:
        # Define pulses - Slice Selective
        ex_1 = cmrseq.seqdefs.excitation.slice_selective_sinc_pulse(system_specs,
                                                                          slice_thickness=Quantity(20, "mm"),
                                                                          flip_angle=FA.to("rad"),
                                                                          pulse_duration=pw,
                                                                          time_bandwidth_product=4,
                                                                          delay=Quantity(0, "ms"),
                                                                          slice_normal=np.array([0., 0., 1.]),
                                                                          slice_position_offset=Quantity(0, "cm"))
        ex_2 = cmrseq.seqdefs.excitation.slice_selective_sinc_pulse(system_specs,
                                                                          slice_thickness=Quantity(20, "mm"),
                                                                          flip_angle=-2*FA.to("rad"),
                                                                          pulse_duration=pw,
                                                                          time_bandwidth_product=4,
                                                                          delay=Quantity(1.56, "ms")-ex_1.end_time,
                                                                          slice_normal=np.array([0., 0., 1.]),
                                                                          slice_position_offset=Quantity(0, "cm"))
        ex_3 = cmrseq.seqdefs.excitation.slice_selective_sinc_pulse(system_specs,
                                                                          slice_thickness=Quantity(20, "mm"),
                                                                          flip_angle=FA.to("rad"),
                                                                          pulse_duration=pw,
                                                                          time_bandwidth_product=4,
                                                                          delay=Quantity(1.56, "ms")-ex_1.end_time,
                                                                          slice_normal=np.array([0., 0., 1.]),
                                                                          slice_position_offset=Quantity(0, "cm"))
        excitation = deepcopy(ex_1)
        excitation.append(ex_2, copy=True)
        excitation.append(ex_3, copy=True)
    else:
        # Define pulses - Non-selective
        ex_1 = cmrseq.bausteine.SincRFPulse(system_specs, FA.to("rad"), duration=pw,
                                        time_bandwidth_product=4)
        ex_2 = cmrseq.bausteine.SincRFPulse(system_specs, -2 * FA.to("rad"), duration=pw,
                                        time_bandwidth_product=4, delay=Quantity(1.56, "ms"))
        ex_3 = cmrseq.bausteine.SincRFPulse(system_specs, FA.to("rad"), duration=pw,
                                        time_bandwidth_product=4, delay=Quantity(2 * 1.56, "ms"))
        excitation = cmrseq.Sequence([ex_1, ex_2, ex_3], system_specs)

                                       # ,[0,0,1],Quantity(0,"T/m"),Quantity(1,"ms"),Quantity(0.2,"ms"))
    # Define readouts
    readout_ref_objects = []
    for te in echo_times:
        ro = cmrseq.seqdefs.readout.epi_cartesian(system_specs, field_of_view=fov,
                                                  matrix_size=matrix_size, adc_duration=adc_duration,
                                                  partial_fourier_lines= pfl)
        adc_center_time = ro.adc_centers[math.ceil(matrix_size[1]/2)-pfl]
        if slice_selective:
            echo_center_shift = ex_2.rf_events[0][0]+ex_1.end_time + te - adc_center_time
        else:
            echo_center_shift = ex_2.rf_events[0] + te - adc_center_time

        if echo_center_shift.m_as("ms") - 3.86 < 0:
            raise ValueError("Echo time to small")
        ro.shift_in_time(system_specs.time_to_raster(echo_center_shift))
        readout_ref_objects.append(ro)

    crusher_grad = cmrseq.bausteine.TrapezoidalGradient.from_area(system_specs=system_specs,
                                                                  orientation=np.array([0., 0., 1.]),
                                                                  area=Quantity(5, "mT/m*ms"))
    crusher_grad.shift_time(readout_ref_objects[-1].end_time)
    rr_shifts = np.cumsum(np.concatenate([Quantity([0.], "ms"), rr_intervals]))
    final_sequence = cmrseq.Sequence([], system_specs)
    for echo_idx, _ in enumerate(rr_intervals):
        seq = cmrseq.Sequence([], system_specs)
        for i in range(echos_per_interval):
            readout_ref_object = readout_ref_objects[echo_idx * 3 + i]
            crusher_grad_seq = cmrseq.Sequence([crusher_grad], system_specs)
            seq.append(excitation + readout_ref_object+crusher_grad_seq, copy=False)
        seq.shift_in_time(rr_shifts[echo_idx])
        final_sequence += seq

    return final_sequence, readout_ref_objects


if __name__ == "__main__":
    echo_times = Quantity(11.6+1.5*np.array([*range(0, 6)]), "ms")
    seq, _ = hyperpol_sequence(fov=Quantity([22, 22], "cm"), matrix_size=(44, 44),
                            echo_times=echo_times, spsp_bandwidth=None,
                            rr_intervals=Quantity([0.9, 1], "s"),
                            adc_duration=Quantity(0.4, "ms"))

    print(seq.duration)
    # f, a = plt.subplots(1, 1, figsize=(12, 4))
    cmrseq.plotting.plot_sequence(seq)
    plt.show()
