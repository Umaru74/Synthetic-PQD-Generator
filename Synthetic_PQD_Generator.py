import numpy as np
import pandas as pd
from pathlib import Path

# ============================== CONFIG ==============================

FUNDAMENTAL_FREQ = 60          # Hz
SAMPLES_PER_CYCLE = 256       # fs = 128*60 = 7680 Hz  (was 256: that gives 2560 samples, the CNN / EMTP converter use 1280)
NUM_CYCLES = 10                # 10 cycles generated for each signal
FS = SAMPLES_PER_CYCLE * FUNDAMENTAL_FREQ  # FS: sampling frequency
DT = 1.0 / FS                  # DT: time step between two consecutive samples
T_END = NUM_CYCLES / FUNDAMENTAL_FREQ      # End of T (s)
T = np.arange(0, T_END, DT)     # time vector, len == SAMPLES_PER_CYCLE*NUM_CYCLES
TCYCLE = 1.0 / FUNDAMENTAL_FREQ            # Tcycle: the duration of one full cycle (s)
W = 2 * np.pi * FUNDAMENTAL_FREQ           # W: angular frequency (radians / second)

# ============================== USER INPUT ==============================

TOTAL_SIGNALS = 48000  # <-- 3000 per class. Random starting phase needs more data than before:
SNR_LEVELS_DB = [20, 30, 40, 50, None]  # None = noiseless
SEED = 74

# --- placement of Sag / Swell / Interruption events (and their combinations) relative to the 10-cycle window ---
WHOLE_WINDOW_FRACTION = 0.3    # a design choice: justify it with a sensitivity study (e.g. 0.2 / 0.3 / 0.4)
DURATION_STEP_CYCLES = 1.0     # 1.0 -> durations of 1, 2 ... 9 cycles; 0.5 -> 0.5, 1.0 ... 9.5 cycles
MIN_OVERLAP_CYCLES = 0.5

# --- harmonics must be observable ---
# Harmonics need a fundamental. Inside an interruption (below 0.1 pu) the generator's harmonics are at most 0.015 pu, which is smaller than
# one step of an int8 input (about 0.012 - 0.016 pu), so a window that is completely inside an interruption cannot show them.
# A window is therefore labeled Interruption+Harmonics only if at least this many cycles of the normal (harmonic-rich) waveform remain visible
# in the 10-cycle window. A window completely, or almost completely, inside an interruption is a plain Interruption.
MIN_VISIBLE_CYCLES = {'Interruption+Harmonics': 2}      # class -> minimum number of visible normal cycles (classes not listed: no limit)

# --- realistic harmonics (found necessary on the real recordings) ---
# [1] builds every harmonic window with the 3rd, 5th and 7th in phase with the fundamental, so all synthetic harmonic windows have the same
# waveform shape. In a real grid every harmonic has its own phase; a CNN trained on in-phase harmonics reads real harmonics as impulsive transients.
HARMONIC_PHASE = 'random'                 # 'random': each harmonic gets a random phase (real grid);  'inphase': as in [1]
HARMONICS_MIN_THD = 0.05                  # harmonic classes: THD of at least this (IEEE Std 519: 5 % for 1-69 kV, 8 % up to 1 kV). None = the ranges of [1] (5-15 % each)
HARMONIC_AMPLITUDE_RANGE = (0.003, 0.15)  # each of the 3rd, 5th, 7th is drawn log-uniformly from this range, redrawn until THD >= HARMONICS_MIN_THD
BACKGROUND_MAX = (0.03, 0.015, 0.015)     # all other windows get a small background distortion (3rd, 5th, 7th: up to these amplitudes), like the real Normal
                                          # windows (THD median 2.0 %, 3rd 1.7 %). None = pure sine. It must stay clearly below HARMONICS_MIN_THD.
HARMONIC_CLASSES = {'Harmonics', 'Sag+Harmonics', 'Swell+Harmonics', 'Flicker+Harmonics', 'Interruption+Harmonics'}

SAVE_FORMAT = 'both'          # 'npz' = float32, compressed, about 6x smaller than the CSV and quick to load; 'csv' = the old text file; 'both'
GENERATION_MODE = 'mixed'  # 'per_snr' -> one file per SNR level
# 'mixed'   -> one combined, shuffled file

OUTPUT_DIR = Path("C:/Users/mfa89/PycharmProjects/ML_Course_Project/Generated_PQD_Dataset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)

# --- auto-compute instances per (class, SNR) combo from TOTAL_SIGNALS ---
_n_combos = 16 * len(SNR_LEVELS_DB)  # 16 classes x however many SNR levels
SAMPLES_PER_CLASS_PER_SNR = max(1, round(TOTAL_SIGNALS / _n_combos))
_actual_total = SAMPLES_PER_CLASS_PER_SNR * _n_combos
print(f"Requested {TOTAL_SIGNALS} signals -> using {SAMPLES_PER_CLASS_PER_SNR} instances "
      f"per (class, SNR) combo -> actual total = {_actual_total} "
      f"(16 classes x {len(SNR_LEVELS_DB)} SNR levels x {SAMPLES_PER_CLASS_PER_SNR})")

# ============================ PARAM RANGES ============================
# One dict per class -> {param_name: (min, max)}. Only params relevant
# to that class are listed; others are unused by create_signal().

PARAM_RANGES = {
    'Normal':                   {},
    'Sag':                      {'alpha': (0.1, 0.9)},
    'Swell':                    {'alpha': (0.1, 0.8)},
    'Interruption':             {'alpha': (0.9, 1.0)},
    'Harmonics':                {'alpha3': (0.05, 0.15), 'alpha5': (0.05, 0.15), 'alpha7': (0.05, 0.15)},
    'Oscillatory transient':    {'alpha': (0.1, 0.8), 'tau': (8e-3, 40e-3), 'Fn': (300, 900)},
    'Impulsive Transient':      {'alpha': (0.1, 0.414)},
    'Sag+Harmonics':            {'alpha': (0.1, 0.9), 'alpha3': (0.05, 0.15), 'alpha5': (0.05, 0.15), 'alpha7': (0.05, 0.15)},
    'Swell+Harmonics':          {'alpha': (0.1, 0.8), 'alpha3': (0.05, 0.15), 'alpha5': (0.05, 0.15), 'alpha7': (0.05, 0.15)},
    'Flicker':                  {'alpha_flicker': (0.1, 0.2), 'beta': (5, 20)},
    'Notch':                    {'k': (0.1, 0.4)},
    'Spike':                    {'k': (0.1, 0.4)},
    'Flicker+Harmonics':        {'alpha_flicker': (0.1, 0.2), 'beta': (5, 20), 'alpha3': (0.05, 0.15), 'alpha5': (0.05, 0.15), 'alpha7': (0.05, 0.15)},
    'Interruption+Harmonics':   {'alpha': (0.9, 1.0), 'alpha3': (0.05, 0.15), 'alpha5': (0.05, 0.15), 'alpha7': (0.05, 0.15)},
    'Flicker+Sag':              {'alpha_flicker': (0.1, 0.2), 'beta': (5, 20), 'alpha': (0.1, 0.9)},
    'Flicker+Swell':            {'alpha_flicker': (0.1, 0.2), 'beta': (5, 20), 'alpha': (0.1, 0.8)},
}

CLASS_NAMES = list(PARAM_RANGES.keys())  # keys() pulls out just "names" of the dictionary such as "normal", "Sag"
assert len(CLASS_NAMES) == 16            # Sanity check, if len is not 16, the code crashes

# Classes with no dependency on a disturbance time window
NO_TIME_WINDOW = {'Normal', 'Harmonics', 'Flicker', 'Flicker+Harmonics'}

# Helper function: it returns the random floating-point number based on the given "lo" and "hi"
def rand_param(lo, hi):
    return rng.uniform(lo, hi)


# function: that decides "when" the disturbances start in the given number of cycles, and "how long" for a cycle
def rand_t1_t2(label):
    min_width = 2 * DT  # At least 2 samples length for the disturbances, min_width = 254 micro second
    # DT: 1/7680 Hz = 130 micro second (one time step)
    # TCYCLE = 1.0 / 60 Hz = 16.67 ms: the duration of one full cycle (s)

    if label in ('Spike', 'Notch'): # This PQD happen at the very beginning of each cycle
        lo = 0.01 * TCYCLE    # Min PQD starting point: 0.01 * TCYCLE = 167 micro seconds for a cycle
        hi = 0.05 * TCYCLE    # Max PQD ending point: 0.05 * TCYCLE = 833 micro seconds for a cycle
        lo = max(lo, min_width)  # lo will be 254 micro seconds (New fixed Min PQD starting point)
        hi = max(hi, lo * 1.5)   # hi will be 834 micro seconds
        window_end = 0.5 * TCYCLE   # meaning that disturbance happens in the first half of each cycle
        hi = min(hi, window_end)  # a single pulse can't need more room than the window itself
        lo = min(lo, hi)  # keep lo<=hi valid even in extreme low-Fs edge cases
        t1 = rng.uniform(0, window_end - hi)  # t1 is picked uniformly between 0 & "window_end - hi"
        t2 = t1 + rng.uniform(lo, hi)

    elif label == 'Oscillatory transient': # This PQD can happen anytime during a cycle
        lo = 0.5 * TCYCLE   # Min PQD Length: 0.5 * TCYCLE = 8.33 ms
        hi = 3 * TCYCLE     # Max PQD Length: 3 * TCYCLE = 50 ms
        lo = max(lo, min_width)     # lo will be 8.33 ms
        t1 = rng.uniform(0, T_END - hi)
        t2 = t1 + rng.uniform(lo, hi)

    elif label == 'Impulsive Transient':  # This PQD can happen anytime during a cycle
        lo = TCYCLE / 20     # Min PQD Starting point: 833 micro seconds
        hi = TCYCLE / 10     # Max PQD Ending point: 1667 micro seconds
        lo = max(lo, min_width)
        hi = max(hi, lo * 1.5)
        t1 = rng.uniform(0, T_END - hi)
        t2 = t1 + rng.uniform(lo, hi)

    else: # Sag, Swell, Interruption and their combinations (see the placement settings at the top)
        if rng.random() < WHOLE_WINDOW_FRACTION:
            t1, t2 = 0.0, T_END                          # the event fills the whole window
        else:
            n_max = int(round((NUM_CYCLES - DURATION_STEP_CYCLES) / DURATION_STEP_CYCLES))
            D = DURATION_STEP_CYCLES * int(rng.integers(1, n_max + 1)) * TCYCLE   # even durations: 1 step ... NUM_CYCLES - 1 step
            ov = min(D, MIN_OVERLAP_CYCLES * TCYCLE)     # at least this much of the event lies inside the window
            t1 = rng.uniform(-D + ov, T_END - ov)        # start time; negative = the event began before the window
            t2 = t1 + D                                  # may be beyond the window end = the event is still on at its end

    return t1, t2


def step_window(t, t1, t2):
    # "astype(float) => converts the True/False array into 1.0 / 0.0.

    # For Sag/Swell/Interruption/Harmonics-combos/Oscillatory Transient/Impulsive Transient, this is called only once
    # For spike / Notch, this function is called repeatedly each cycle
    return ((t >= t1) & (t < t2)).astype(float)


def create_signal(label, params):
    t = T   # time vector for each sample point
    w = W   # Angular frequency => w = 2 * pi * 60 Hz = 377 radians / second
    min_width = 2 * DT      # DT: One time step, min_width: Minimum PQDs length (at least 2 sample time)

    # ---- RANDOM STARTING PHASE ------------------------------------------------------------
    # A real recording can start anywhere in the cycle, so the 60 Hz carrier AND its harmonics are shifted
    t_shift = rng.uniform(0, TCYCLE)
    ts = t + t_shift                       # "carrier time": use ts inside every sin(w*...) below
    phi_f = rng.uniform(0, 2 * np.pi)      # random phase of the flicker ripple
    if HARMONIC_PHASE == 'random':         # phase of the 3rd, 5th and 7th relative to the fundamental
        p3, p5, p7 = rng.uniform(0, 2 * np.pi, 3)
    else:
        p3 = p5 = p7 = 0.0
    if BACKGROUND_MAX is not None and label not in HARMONIC_CLASSES:      # a real grid is never a pure sine
        b3, b5, b7 = (rng.uniform(0, m) for m in BACKGROUND_MAX)
        carrier = np.sin(w * ts) + b3 * np.sin(3 * w * ts + p3) + b5 * np.sin(5 * w * ts + p5) + b7 * np.sin(7 * w * ts + p7)
    else:
        carrier = np.sin(w * ts)
    # ----------------------------------------------------------------------------------------

    # this checks if this type of PQD requires a disturbance time window (t1 and t2)
    needs_window = label not in NO_TIME_WINDOW

    # Initializes both to None
    t1 = t2 = None

    # If a PQD requires time window
    if needs_window:
        t1, t2 = rand_t1_t2(label)
        need = MIN_VISIBLE_CYCLES.get(label, 0) * TCYCLE
        if need > 0:                                       # draw the placement again until enough normal cycles remain visible
            for _ in range(10000):
                inside = max(0.0, min(t2, T_END) - max(t1, 0.0))
                if T_END - inside >= need:
                    break
                t1, t2 = rand_t1_t2(label)
            else:
                raise RuntimeError(f"no placement leaves {MIN_VISIBLE_CYCLES[label]} visible cycles for {label}")
        win = step_window(t, t1, t2)

    if label == 'Normal':
        y = carrier

    elif label == 'Sag':
        y = (1 - params['alpha'] * win) * carrier

    elif label == 'Swell':
        y = (1 + params['alpha'] * win) * carrier

    elif label == 'Interruption':
        y = (1 - params['alpha'] * win) * carrier

    elif label == 'Harmonics':
        a3, a5, a7 = params['alpha3'], params['alpha5'], params['alpha7']
        a1 = np.sqrt(max(0, 1 - a3**2 - a5**2 - a7**2))
        y = a1*np.sin(w*ts) + a3*np.sin(3*w*ts + p3) + a5*np.sin(5*w*ts + p5) + a7*np.sin(7*w*ts + p7)

    elif label == 'Oscillatory transient':
        tau, Fn, alpha = params['tau'], params['Fn'], params['alpha']
        env = np.where(t >= t1, np.exp(-(t - t1) / tau), 0.0)
        y = carrier + alpha * win * env * np.sin(2*np.pi*Fn*(t - t1))

    elif label == 'Impulsive Transient':
        y = (1 - params['alpha'] * win) * carrier

    elif label == 'Sag+Harmonics':
        a3, a5, a7 = params['alpha3'], params['alpha5'], params['alpha7']
        a1 = np.sqrt(max(0, 1 - a3**2 - a5**2 - a7**2))
        base = a1*np.sin(w*ts) + a3*np.sin(3*w*ts + p3) + a5*np.sin(5*w*ts + p5) + a7*np.sin(7*w*ts + p7)
        y = (1 - params['alpha'] * win) * base

    elif label == 'Swell+Harmonics':
        a3, a5, a7 = params['alpha3'], params['alpha5'], params['alpha7']
        a1 = np.sqrt(max(0, 1 - a3**2 - a5**2 - a7**2))
        base = a1*np.sin(w*ts) + a3*np.sin(3*w*ts + p3) + a5*np.sin(5*w*ts + p5) + a7*np.sin(7*w*ts + p7)
        y = (1 + params['alpha'] * win) * base

    elif label == 'Flicker':
        y = (1 + params['alpha_flicker'] * np.sin(params['beta']*2*np.pi*t + phi_f)) * carrier

    elif label in ('Spike', 'Notch'):
        k = params['k']     # 0.1 < K < 0.4
        pulse = np.zeros_like(t)
        # the pulses are tied to the carrier's own cycles, so they use ts (not t); one extra cycle
        # because the shifted window ends part-way into the next cycle
        for n in range(NUM_CYCLES + 1):
            pulse += step_window(ts, t1 + TCYCLE*n, t2 + TCYCLE*n)
        pulse *= k      # Multiply pulse by pulse amplitude "k"
        sgn = np.sign(np.sin(w * ts))   # sign of the carrier
        y = carrier + sgn*pulse if label == 'Spike' else carrier - sgn*pulse

    elif label == 'Flicker+Harmonics':
        a3, a5, a7 = params['alpha3'], params['alpha5'], params['alpha7']
        a1 = np.sqrt(max(0, 1 - a3**2 - a5**2 - a7**2))
        base = a1*np.sin(w*ts) + a3*np.sin(3*w*ts + p3) + a5*np.sin(5*w*ts + p5) + a7*np.sin(7*w*ts + p7)
        y = (1 + params['alpha_flicker']*np.sin(params['beta']*2*np.pi*t + phi_f)) * base

    elif label == 'Interruption+Harmonics':
        a3, a5, a7 = params['alpha3'], params['alpha5'], params['alpha7']
        a1 = np.sqrt(max(0, 1 - a3**2 - a5**2 - a7**2))
        base = a1*np.sin(w*ts) + a3*np.sin(3*w*ts + p3) + a5*np.sin(5*w*ts + p5) + a7*np.sin(7*w*ts + p7)
        y = (1 - params['alpha'] * win) * base

    elif label in ('Flicker+Sag', 'Flicker+Swell'):
        sign = -1 if label == 'Flicker+Sag' else 1
        y = (1 + params['alpha_flicker']*np.sin(params['beta']*2*np.pi*t + phi_f)) * carrier * \
            (1 + sign*params['alpha']*win)

    else:
        raise ValueError(f"Unknown label: {label}")

    return y


def add_noise(y, snr_db):
    if snr_db is None:
        return y
    sig_power = np.mean(y**2)     # Computes the average power of the clean signal
    noise_power = sig_power / (10**(snr_db/10))     #
    noise = rng.normal(0, np.sqrt(noise_power), size=y.shape)
    return y + noise


def draw_params(label):
    """The parameters of one window. The harmonic classes get log-uniform 3rd/5th/7th amplitudes with THD >= HARMONICS_MIN_THD."""
    params = {name: rand_param(lo, hi) for name, (lo, hi) in PARAM_RANGES[label].items()}
    if label in HARMONIC_CLASSES and HARMONICS_MIN_THD is not None:
        lo, hi = np.log(HARMONIC_AMPLITUDE_RANGE[0]), np.log(HARMONIC_AMPLITUDE_RANGE[1])
        for _ in range(10000):
            a = np.exp(rng.uniform(lo, hi, 3))
            if np.sqrt(np.sum(a ** 2)) >= HARMONICS_MIN_THD:
                break
        params['alpha3'], params['alpha5'], params['alpha7'] = a
    return params


# ============================== GENERATE ==============================

def generate_balanced_grid():
    """Generates every (class, SNR) combination with exactly
    SAMPLES_PER_CLASS_PER_SNR instances each - the balanced 'grid'
    that both modes are built from."""
    rows, labels, snrs = [], [], []
    for snr in SNR_LEVELS_DB:
        for label in CLASS_NAMES:
            for _ in range(SAMPLES_PER_CLASS_PER_SNR):
                params = draw_params(label)
                y = create_signal(label, params)
                y = add_noise(y, snr)
                rows.append(y)
                labels.append(label)
                snrs.append(snr if snr is not None else np.inf)
    return np.array(rows), np.array(labels), np.array(snrs)


def save_dataset(X, y_labels, snr_arr, tag):
    base = f"pqd_dataset_{tag}_{FUNDAMENTAL_FREQ}Hz_{SAMPLES_PER_CYCLE}SamplesPerCycle_For_Test"
    if SAVE_FORMAT in ('npz', 'both'):
        path = OUTPUT_DIR / f"{base}.npz"
        np.savez_compressed(path, X=X.astype(np.float32), y=y_labels, snr=snr_arr)
        back = np.load(path)                                        # read it back: the saved file must have the rows that were generated
        assert back['X'].shape == X.shape and len(back['y']) == len(y_labels), "the saved .npz does not match the generated data"
        print(f"  Saved and verified: {path.name}  ({back['X'].shape[0]} rows, {path.stat().st_size / 1e6:.0f} MB)")
    if SAVE_FORMAT in ('csv', 'both'):
        path = OUTPUT_DIR / f"{base}.csv"
        df = pd.DataFrame(X, columns=[f"sample_{i}" for i in range(X.shape[1])])
        df['label'] = y_labels
        df['snr_db'] = snr_arr
        df.to_csv(path, index=False)
        with open(path, 'rb') as f:                                  # count the lines: a cut-off file has fewer rows than were generated
            n_rows = sum(1 for _ in f) - 1
        assert n_rows == len(X), f"the CSV has {n_rows} rows, but {len(X)} were generated"
        print(f"  Saved and verified: {path.name}  ({n_rows} rows, {path.stat().st_size / 1e6:.0f} MB)")


if GENERATION_MODE == 'per_snr':
    for snr in SNR_LEVELS_DB:
        rows, labels, snrs = [], [], []
        for label in CLASS_NAMES:
            for _ in range(SAMPLES_PER_CLASS_PER_SNR):
                params = draw_params(label)
                y = create_signal(label, params)
                y = add_noise(y, snr)
                rows.append(y)
                labels.append(label)
                snrs.append(snr if snr is not None else np.inf)

        X = np.array(rows)
        y_labels = np.array(labels)
        snr_arr = np.array(snrs)
        snr_tag = "noiseless" if snr is None else f"snr{snr:g}db"

        print(f"SNR={snr_tag}: generated {X.shape[0]} signals "
              f"({len(CLASS_NAMES)} classes x {SAMPLES_PER_CLASS_PER_SNR} instances)")
        save_dataset(X, y_labels, snr_arr, snr_tag)

    print(f"\nAll {len(SNR_LEVELS_DB)} SNR-specific datasets saved to {OUTPUT_DIR}")

elif GENERATION_MODE == 'mixed':
    X, y_labels, snr_arr = generate_balanced_grid()

    shuffle_idx = rng.permutation(len(X))
    X = X[shuffle_idx]
    y_labels = y_labels[shuffle_idx]
    snr_arr = snr_arr[shuffle_idx]

    total = len(X)
    print(f"Mixed dataset: {total} signals total "
          f"({len(CLASS_NAMES)} classes x {len(SNR_LEVELS_DB)} SNR levels x {SAMPLES_PER_CLASS_PER_SNR} instances)")

    import collections

    class_counts = collections.Counter(y_labels)
    snr_counts = collections.Counter(snr_arr)
    print(f"  Per-class counts: {dict(class_counts)}")
    print(f"  Per-SNR counts:   {dict(snr_counts)}")

    save_dataset(X, y_labels, snr_arr, "mixed")

else:
    raise ValueError(f"Unknown GENERATION_MODE: {GENERATION_MODE}")
