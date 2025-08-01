import torch
def short_time_energy(
    signal,
    sample_rate,
    frame_duration=0.05,
):
    """
    Compute short-time energy.

    Args:
        signal (torch.Tensor): Input signal, shape [N].
        sample_rate (int): Sampling rate.
        frame_duration (float): Frame duration in seconds, default is 0.05 seconds.
        hop_duration (float): Frame hop length in seconds, default is 0.02 seconds.

    Returns:
        torch.Tensor: Short-time energy, shape [num_frames].
    """
    frame_size = int(frame_duration * sample_rate)

    # Use unfold to segment the signal into frames
    frames = signal.unfold(0, frame_size, frame_size)  # Shape: [num_frames, frame_size]

    # Compute energy for each frame
    energy = (frames**2).sum(dim=1)  # Shape: [num_frames]
    return energy


def short_time_zero_crossing_rate(
    signal,
    sample_rate,
    frame_duration=0.05,
):
    """
    Compute short-time zero-crossing rate.

    Args:
        signal (torch.Tensor): Input signal, shape [N].
        sample_rate (int): Sampling rate.
        frame_duration (float): Frame duration in seconds, default is 0.05 seconds.
        hop_duration (float): Frame hop length in seconds, default is 0.02 seconds.

    Returns:
        torch.Tensor: Short-time zero-crossing rate, shape [num_frames].
    """
    frame_size = int(frame_duration * sample_rate)

    # Use unfold to segment the signal into frames
    frames = signal.unfold(0, frame_size, frame_size)  # Shape: [num_frames, frame_size]

    # Compute the number of zero-crossings for adjacent samples
    zero_crossings = (
        ((frames[:, :-1] * frames[:, 1:]) < 0).float().sum(dim=1)
    )  # Shape: [num_frames]
    return zero_crossings


def double_threshold_vad(
    signal,
    sample_rate,
    energy_thresholds=(0.2, 1.0),
    zcr_threshold=10,
    frame_duration=0.05,
):
    """
    Voice Activity Detection (VAD) using double-threshold method.

    Args:
        signal (torch.Tensor): Input signal, shape [N].
        sample_rate (int): Sampling rate.
        energy_thresholds (tuple): Energy thresholds (low_threshold, high_threshold), default is (0.1, 0.5).
        zcr_threshold (float): Zero-crossing rate threshold, default is 30.
        frame_duration (float): Frame duration in seconds, default is 0.05 seconds.
        hop_duration (float): Frame hop length in seconds, default is 0.02 seconds.

    Returns:
        torch.Tensor: Detection result, shape [num_frames], values are 0 or 1.
    """
    # Compute short-time energy and zero-crossing rate
    energy = short_time_energy(signal, sample_rate, frame_duration)
    zcr = short_time_zero_crossing_rate(signal, sample_rate, frame_duration)

    # Initialize detection result
    num_frames = energy.shape[0]
    vad_result = torch.zeros(num_frames)

    low_threshold, high_threshold = energy_thresholds

    # Double-threshold logic
    for i in range(num_frames):
        if energy[i] > high_threshold or (
            energy[i] > low_threshold and zcr[i] > zcr_threshold
        ):
            vad_result[i] = 1

    return vad_result
