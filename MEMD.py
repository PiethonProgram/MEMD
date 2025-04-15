from calculations import *


def validate_memd_input(signal, n_dir, stop_crit, stop_vec, n_iter, max_imf, e_thresh):
    """
        Validates and preprocesses inputs for the multi-dimensional EMD (MEMD) process.

        Parameters:
          signal (numpy.array): The multi-channel signal to decompose.
          n_dir (int): Number of direction vectors for envelope calculation. Must be >= 6.
          stop_crit (str): Stopping criterion used in the sifting process. Options: 'stop', 'fix_h', 'e_diff'.
          stop_vec (list/tuple/np.array): List of three threshold values controlling the stopping criterion.
          n_iter (int): Number of sifting iterations if using the 'fix_h' stopping criterion.
          max_imf (int): Maximum number of Intrinsic Mode Functions (IMFs) to extract. Must be a positive integer.
          e_thresh (float): Energy threshold for the stopping criterion 'e_diff'.

        Returns:
          tuple: (signal, n_dir, stop_crit, stop_vec, n_iter, max_imf, e_thresh, N_dim, N)
                 where N_dim is the number of channels and N is the number of samples.

        Exits:
          Terminates the program if any validation fails.
    """

    if len(signal) == 0:
        sys.exit('empty dataset. No Data found')

    if signal.shape[0] < signal.shape[1]:
        signal = signal.T

    N_dim = signal.shape[1]

    if N_dim < 3:
        sys.exit('Function only processes the signal having more than 3 channels.')
    N = signal.shape[0]

    if not isinstance(n_dir, int) or n_dir < 6:
        sys.exit('invalid num_dir. num_dir should be an integer greater than or equal to 6.')
    if not isinstance(stop_crit, str) or (stop_crit not in ['stop', 'fix_h', 'e_diff']):
        sys.exit('invalid stop_criteria. stop_criteria should be either fix_h, stop or e_diff')
    if not isinstance(stop_vec, (list, tuple, np.ndarray)) or any(
            x for x in stop_vec if not isinstance(x, (int, float, complex))):
        sys.exit(
            'invalid stop_vector. stop_vector should be a list with three elements e.g. default is [0.75,0.75,0.75]')
    if stop_crit == 'fix_h' and (not isinstance(n_iter, int) or n_iter < 0):
        sys.exit('invalid stop_count. stop_count should be a nonnegative integer.')

    if not isinstance(max_imf, int) or max_imf < 1:
        sys.exit('invalid max_imf. max_imf should be a positive integer')

    return signal, n_dir, stop_crit, stop_vec, n_iter, max_imf, e_thresh, N_dim, N


def memd(signal, n_dir=50, stop_crit='stop', stop_vec=(0.05, 0.5, 0.05), n_iter=3, max_imf=100, e_thresh=1e-3):
    """
        Applies multi-dimensional empirical mode decomposition (MEMD) on a multi-channel signal.

        Process:
          - Validates and pre-processes the input signal.
          - Sets up direction vectors for computing signal envelopes.
          - Iteratively extracts intrinsic mode functions (IMFs) via sifting until the stopping condition is reached
            or until the maximum number of IMFs is extracted.
          - Supports various stopping criteria:
              'stop'   : Standard threshold-based stop.
              'fix_h'  : Fixed number of sifting iterations.
              'e_diff' : Energy-difference based stop (uses previous IMF for comparison).

        Parameters:
          signal (numpy.array): Input multi-channel signal.
          n_dir (int): Number of direction vectors for envelope estimation.
          stop_crit (str): Selected stopping criterion ('stop', 'fix_h', or 'e_diff').
          stop_vec (tuple): Threshold values that control the stopping criterion.
          n_iter (int): Iteration limit for the 'fix_h' stopping criterion.
          max_imf (int): Maximum number of IMFs to extract.
          e_thresh (float): Threshold for energy difference in the 'e_diff' criterion.

        Returns:
          numpy.array: A multi-dimensional array containing the extracted IMFs and the residual component.
                       The output is transposed to have dimensions (channels, IMF_number, samples).
    """

    signal, n_dir, stop_crit, stop_vec, n_iter, max_imf, e_thresh, N_dim,  N = (
        validate_memd_input(signal, n_dir, stop_crit, stop_vec, n_iter, max_imf, e_thresh))

    base = [-n_dir]

    if N_dim == 3:
        base = np.array([-n_dir, 2])
        seq = np.zeros((n_dir, N_dim - 1))
        seq[:, 0] = hamm(n_dir, base[0])
        seq[:, 1] = hamm(n_dir, base[1])
    else:
        prm = nth_prime(N_dim - 1)
        for itr in range(1, N_dim):
            base.append(prm[itr - 1])
        seq = np.zeros((n_dir, N_dim))
        for it in range(N_dim):
            seq[:, it] = hamm(n_dir, base[it])

    t = np.arange(1, N + 1)
    nbit = 0
    MAXITERATIONS = 1000
    sd, sd2, tol = stop_vec[0], stop_vec[1], stop_vec[2] if stop_crit == 'stop' else (None, None, None)
    stp_cnt = n_iter if stop_crit == 'fix_h' else None
    e_thresh = e_thresh if stop_crit == 'e_diff' else None
    r = signal
    n_imf = 1
    imfs = []
    prev_imf = None

    while not stop_emd(r, seq, n_dir, N_dim) and n_imf <= max_imf:
        m = r.copy()
        if stop_crit == 'stop':
            stop_sift, env_mean = stop(m, t, sd, sd2, tol, seq, n_dir, N, N_dim)
        elif stop_crit == 'fix_h':
            counter = 0
            stop_sift, env_mean, counter = fix(m, t, seq, n_dir, stp_cnt, counter, N, N_dim)
        elif stop_crit == 'e_diff':
            # If there is a previous IMF, calculate the energy difference
            if prev_imf is not None:
                stop_sift, env_mean = e_diff(prev_imf, m, t, seq, n_dir, N, N_dim, e_thresh)
            else:
                # For the first IMF, use a regular stopping criterion
                stop_sift, env_mean = stop(m, t, sd, sd2, tol, seq, n_dir, N, N_dim)

        if np.max(np.abs(m)) < 1e-10 * np.max(np.abs(signal)):
            if not stop_sift:
                print('emd:warning : forced stop of EMD : amplitude too small')
            else:
                print('forced stop of EMD : amplitude too small')
            break

        while not stop_sift and nbit < MAXITERATIONS:
            m -= env_mean
            if stop_crit == 'stop':
                stop_sift, env_mean = stop(m, t, sd, sd2, tol, seq, n_dir, N, N_dim)
            elif stop_crit == 'fix_h':
                stop_sift, env_mean, counter = fix(m, t, seq, n_dir, stp_cnt, counter, N, N_dim)
            elif stop_crit == 'e_diff':
                if prev_imf is not None:
                    stop_sift, env_mean = e_diff(prev_imf, m, t, seq, n_dir, N, N_dim, e_thresh)
                else:
                    stop_sift, env_mean = stop(m, t, sd, sd2, tol, seq, n_dir, N, N_dim)
            nbit += 1
            if nbit == (MAXITERATIONS - 1) and nbit > 100:
                print('emd:warning : forced stop of sifting : too many iterations')

        imfs.append(m.T)
        n_imf += 1
        r = r - m
        nbit = 0
        # prev_imf = m  # Update prev_imf after extracting the IMF

    imfs.append(r.T)

    return np.asarray(imfs).transpose(1, 0, 2)

# FINAL CHANGES 
