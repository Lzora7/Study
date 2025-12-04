import numpy as np
from additive import adsr_envelope
from pyarrow import DurationScalar

def ks_note(sr, note, duration, decay):
    """
    Generates a sound wave using the Karplus-Strong algorithm.

    Parameters:
    sr (int): The sample rate.
    note (float): The note value, where 0 corresponds to A4 (440 Hz).
    duration (float): The duration of the note in seconds.
    decay (float): The decay factor for the algorithm, controlling the damping of the sound.

    Returns:
    numpy.ndarray: The generated sound wave as a numpy array.
    """
    
    # Calculate the frequency of the note
    freq = 440 * 2 ** (note / 12.0)
    
    # Calculate the length of the delay line
    T = int(sr / freq)
    
    # Initialize the delay line with random noise
    delay_line = np.random.rand(T)

    # Initialize the output buffer
    #╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    output = []

    # Fill the output buffer using the Karplus-Strong algorithm
    #╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ'

    for i in range(int(sr*duration)): # ход по каждой точке дискретизации во всей длительности
        output.append(delay_line[0])

        cur_out = decay * ((delay_line[0] + delay_line[1]) / 2)
        delay_line = np.roll(delay_line, -1) # первый элемент стал последним (сдвиг)

        delay_line[-1] = cur_out

    return np.array(output)

def make_melody(filename, sixteenth_len, sr, note_function):
    """
    Parameters
    ----------
    filename: string
        Path to file containing the tune.  Consists of
        rows of <note number> <note duration>, where
        the note number 0 is a 440hz concert A, and the
        note duration is in factors of 16th notes
    sixteenth_len : float
        Duration of a sixteenth note in seconds. This parameter is used to convert the note durations
        from the file into actual time durations.
    sr: int
        Sample rate
    note_function : function
        Function to generate audio samples for a single note. It should take three parameters:
        sr (sample rate), note (note number or frequency), and duration (duration of the note in seconds),
        and return a NumPy array of audio samples.

    Returns
    -------
    np.ndarray
        Array of audio samples representing the generated melody.
    """
    melody = np.loadtxt(filename)
    notes = melody[:, 0] 
    durations = sixteenth_len * melody[:, 1] 
    
    
    # Initialize an empty list to store the audio samples
    audio_samples = []
    
    # Generate audio samples for each note and append to the list
    #╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    for i, note in enumerate(notes):
        if np.isnan(note):
            note_sound = np.zeros(int(sr*durations[i]))
        else:
            note_sound = note_function(sr, note, durations[i])
            audio_samples.append(note_sound)
    
    # Concatenate all the audio samples into a single array
    #╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    out = np.concatenate(audio_samples)

    return out


def fm_note(sr, note, duration, ratio=2, I=2, 
                  envelope=lambda N, sr: np.ones(N),
                  amplitude=lambda N, sr: np.ones(N)):
    """
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio
    ratio: float
        Ratio of modulation frequency to carrier frequency
    I: float
        Modulation index (ratio of peak frequency deviation to
        modulation frequency)
    envelope: function (N, sr) -> ndarray(N)
        A function for generating an envelope profile
    amplitude: function (N, sr) -> ndarray(N)
        A function for generating a time-varying amplitude

    Returns
    -------
    ndarray(N): Audio samples for this note
    """
    # Calculate the number of samples
    #╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    num_samples = int(sr * duration)
    
    # Calculate the carrier frequency (fc)
    fc = 440 * 2 ** (note / 12.0) # - несущая частота
    
    # Calculate the modulation frequency (fm)
    #╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    fm = fc * ratio
    
    # Generate the time array
    #╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    time_array = np.arange(num_samples) / sr
    
    # Generate the envelopes for amplitude and modulation index
    #╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    ampl = amplitude(num_samples, sr)
    modul = envelope(num_samples, sr)
    
    # Generate the FM waveform
    #╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    fm_wave = ampl * np.sin(
       2*np.pi * fc * time_array + \
       I*modul*np.sin(2*np.pi * fm * time_array)
    )

    return fm_wave


def exp_env(N, sr, mu=3):
    """
    Make an exponential envelope
    Parameters
    ----------
    N: int
        Number of samples
    sr: int
        Sample rate
    mu: float
        Exponential decay rate: e^{-mu*t}

    Returns
    -------
    ndarray(N): Envelope samples
    """
    return np.exp(-mu*np.arange(N)/sr)


def fm_string_note(sr, note, duration, mu=3):
    """
    Make a string of a particular length
    using FM synthesis
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio
    mu: float
        The decay rate of the note
    
    Returns
    -------
    ndarray(N): Audio samples for this note
    """
    envelope = lambda N, sr: exp_env(N, sr, mu)
    return fm_note(sr, note, duration,
                ratio = 1, I = 8, envelope = envelope,
                amplitude = envelope)


def fm_el_guitar_note(sr, note, duration, mu=3):
    """
    Make an electric guitar string of a particular length by
    passing along the parameters to fm_plucked_string note
    and then turning the samples into a square wave

    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio
    mu: float
        The decay rate of the note
    
    Return
    ------
    ndarray(N): Audio samples for this note
    """
    # Generate the plucked string sound
    #╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    base_sound = fm_string_note(sr, note, duration, mu=3)

    # Convert the plucked string sound to a square wave
    #╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    guitar_sound = np.sign(base_sound)

    return guitar_sound


def fm_bell_note(sr, note, duration):
    """
    Make a bell note of a particular length
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio
    
    Returns
    -------
    ndarray(N): Audio samples for this note
    """
    envelope = lambda N, sr: exp_env(N, sr, 0.8)
    return fm_note(sr, note, duration,
                ratio = 1.4, I = 2, envelope = envelope,
                amplitude = envelope)


def brass_env(N, sr):
    """
    Make the brass ADSR envelope from Chowning's paper

    Parameters
    ----------
    N : int
        The number of samples in the envelope.
    sr : int
        The sample rate, which is the number of samples per second.

    Returns
    -------
    ndarray
        An array of length N containing the envelope samples.
    """
    
    duration = N / sr
    sustain_level = 0.8 

    if duration < 0.3:
        attack_prop = 1/3
        decay_prop = 1/3 
        release_prop = 1/3
       
    else:
        attack_prop = 0.1 / duration
        decay_prop = 0.1 / duration
        release_prop = 0.1 / duration

    envelope = adsr_envelope(N, 
                            attack=attack_prop, 
                            decay=decay_prop, 
                            sustain=sustain_level, 
                            release=release_prop
                            )

    return envelope


def fm_brass_note(sr, note, duration):
    """
    Make a brass note of a particular length
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number.  0 is 440hz concert A
    duration: float
        Seconds of audio
    
    Return
    ------
    ndarray(N): Audio samples for this note
    """
    #╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ

    envelope = lambda N, sr: brass_env(N, sr)
    return fm_note(sr, note, duration,
                ratio = 1, I = 10, envelope = envelope,
                amplitude = envelope)



def drum_env(N, sr):
    """
    Make a drum envelope, according to Chowning's paper
    Parameters
    ----------
    N: int
        Number of samples
    sr: int
        Sample rate

    Returns
    -------
    ndarray(N): Envelope samples
    """
    #╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    
    mu = 85

    t = np.arange(N) / sr
    envelope = (t**2) * np.exp(-mu * t)
    envelope = envelope / np.max(envelope)

    return envelope


def fm_drum_sound(sr, note, duration, fixed_note=-14):
    """
    Make what Chowning calls a "drum-like sound"
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number (which is ignored)
    duration: float
        Seconds of audio
    fixed_note: int
        Note number of the fixed note for this drum
    
    Returns
    ------
    ndarray(N): Audio samples for this drum hit
    """
    #╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    
    envelope = lambda N, sr: drum_env(N, sr)
    return fm_note(sr, fixed_note, duration,
                ratio = 1.4, I = 2, envelope = envelope,
                amplitude = envelope)


def snare_drum_sound(sr, note, duration):
    """
    Make a snare drum sound by shaping noise
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number (which is ignored)
    duration: float
        Seconds of audio
    
    Returns
    -------
    ndarray(N): Audio samples for this drum hit
    """
    #╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    N = int(sr * duration)

    noise = np.random.rand(N)
    sound = noise * drum_env(N, sr)

    return sound


def wood_drum_env(N, sr):
    """
    Make the wood-drum envelope from Chowning's paper
    Parameters
    ----------
    N: int
        Number of samples
    sr: int
        Sample rate

    Returns
    -------
    ndarray(N): Envelope samples
    """
    #╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ

    decay_time = 0.008 # сек
    
    # кол сэмплов для decay
    decay_samples = int(sr * decay_time)
    
    envelope = np.zeros(N)
    
    if decay_samples > 0:
        envelope[:decay_samples] = np.linspace(1, 0, decay_samples)
    
    return envelope


def fm_wood_drum_sound(sr, note, duration, fixed_note=-14):
    """
    Make what Chowning calls a "wood drum sound"
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number (which is ignored)
    duration: float
        Seconds of audio
    fixed_note: int
        Note number of the fixed note for this drum
    
    Returns
    -------
    ndarray(N): Audio samples for this drum hit
    """
    #╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    
    envelope = lambda N, sr: wood_drum_env(N, sr)
    return fm_note(sr, fixed_note, duration,
                ratio = 1.4, I = 10, envelope = envelope,
                amplitude = envelope)


def dirty_bass_env(N, sr):
    """
    Make the "dirty bass" envelope 
    
    Parameters
    ----------
    N: int
        Number of samples
    sr: int
        Sample rate
    
    Returns
    -------
    ndarray(N): Envelope samples
    """
    #╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    mu = 5

    func_change_prop = 0.5 # пропорция разных функций
    part_N = N * func_change_prop # сколько N в доле

    t1 = np.arange(part_N) / sr
    t2 = np.arange(N - part_N) / sr

    envelope_part_1 = np.exp(-mu * t1)
    envelope_part_2 = np.exp(-mu * t2)

    envelope = np.concatenate([envelope_part_1, envelope_part_2])
    envelope = envelope / np.max(envelope)
    
    return envelope


def fm_dirty_bass_note(sr, note, duration):
    """
    Make a "dirty bass" note
    
    Parameters
    ----------
    sr: int
        Sample rate
    note: int
        Note number (which is ignored)
    duration: float
        Seconds of audio
    
    Returns
    -------
    ndarray(N): Audio samples for this drum hit
    """
    #╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    
    envelope = lambda N, sr: dirty_bass_env(N, sr)
    return fm_note(sr, note, duration,
                ratio = 1, I = 18, envelope = envelope,
                amplitude = envelope)