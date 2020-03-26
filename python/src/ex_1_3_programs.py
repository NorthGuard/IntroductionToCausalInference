import numpy as np
from tqdm import tqdm


# Settings
_var_x = 1
_var_y = 4
_rho_xy = 0.7
_mu_y = 2
_mu_x = 1
_mu_z = 1
_var_z = 2

_plot_limit_stds = 3

####


# Standard deviations
_std_x = np.sqrt(_var_x)
_std_y = np.sqrt(_var_y)
_std_z = np.sqrt(_var_z)

# Mixing coefficients - x-into-y etc.
a_xy = _std_y * _rho_xy / _std_x
a_yx = _std_x * _rho_xy / _std_y
a_zx = _std_x * np.sqrt(_rho_xy) / _std_z
a_zy = _std_y * np.sqrt(_rho_xy) / _std_z

# Noise variances
_noise_var_yx = _var_x * (1 - np.power(_rho_xy, 2))
_noise_var_xy = _var_y * (1 - np.power(_rho_xy, 2))
_noise_var_zx = _var_x * (1 - _rho_xy)
_noise_var_zy = _var_y * (1 - _rho_xy)

# Noise standard deviations
_noise_std_yx = np.sqrt(_noise_var_yx)
_noise_std_xy = np.sqrt(_noise_var_xy)
_noise_std_zx = np.sqrt(_noise_var_zx)
_noise_std_zy = np.sqrt(_noise_var_zy)

# Noise offsets - x-into-y etc.
_noise_offset_xy = _mu_y - _mu_x * a_xy
_noise_offset_yx = _mu_x - _mu_y * a_yx
_noise_offset_zx = _mu_x - _mu_z * a_zx
_noise_offset_zy = _mu_y - _mu_z * a_zy

# Compute limits
x_limits = (_mu_x - _plot_limit_stds * _std_x, _mu_x + _plot_limit_stds * _std_x)
y_limits = (_mu_y - _plot_limit_stds * _std_y, _mu_y + _plot_limit_stds * _std_y)
z_limits = (_mu_z - _plot_limit_stds * _std_z, _mu_z + _plot_limit_stds * _std_z)


def program_2(n_samples, x=None, y=None, _z=None, __seed=6):
    # Ensure same points if more are added
    np.random.seed(__seed)
    norm_samples = np.random.randn(n_samples, 2).T

    # Make x or intervene
    if x is None:
        # Make normal
        x = _mu_x + _std_x * norm_samples[0, :]
    else:
        # Disallow extreme stuff and force
        x = np.clip(x, a_min=x_limits[0], a_max=x_limits[1])
        x = np.ones((n_samples,)) * x

    # Make y or intervene
    if y is None:
        # Noise distribution
        N_y = _noise_std_xy * norm_samples[1, :] + _noise_offset_xy

        # Y = X * a + N_y
        y = x * a_xy + N_y
    else:
        # Disallow extreme stuff and force
        y = np.clip(y, a_min=y_limits[0], a_max=y_limits[1])
        y = np.ones((n_samples,)) * y

    # Stack data and return
    out = np.stack((x, y), axis=1)
    return out


def program_3(n_samples, x=None, y=None, _z=None, __seed=2):
    # Ensure same points if more are added
    np.random.seed(__seed)
    norm_samples = np.random.randn(n_samples, 2).T

    # Make y or intervene
    if y is None:
        # Make normal
        y = _mu_y + _std_y * norm_samples[0, :]
    else:
        # Disallow extreme stuff and force
        y = np.clip(y, a_min=y_limits[0], a_max=y_limits[1])
        y = np.ones((n_samples,)) * y

    # Make x or intervene
    if x is None:
        # Noise distribution
        N_x = _noise_std_yx * norm_samples[1, :] + _noise_offset_yx

        # X = Y * a + N_x
        x = y * a_yx + N_x
    else:
        # Disallow extreme stuff and force
        x = np.clip(x, a_min=x_limits[0], a_max=x_limits[1])
        x = np.ones((n_samples,)) * x

    # Stack data and return
    out = np.stack((x, y), axis=1)
    return out


def program_1(n_samples, x=None, y=None, _z=None, __seed=3):
    # Ensure same points if more are added
    np.random.seed(__seed)
    norm_samples = np.random.randn(n_samples, 3).T

    # Make confounder
    if _z is None:
        # Make normal
        z = _std_z * norm_samples[0, :] + _mu_z
    else:
        # Disallow extreme stuff and force
        z = np.clip(_z, a_min=z_limits[0], a_max=z_limits[1])
        z = np.ones((n_samples,)) * z

    # Make y or intervene
    if y is None:
        # Noise distribution
        N_y = _noise_std_zy * norm_samples[1, :] + _noise_offset_zy

        # Y = Z * a + N_y
        y = z * a_zy + N_y
    else:
        # Disallow extreme stuff and force
        y = np.clip(y, a_min=y_limits[0], a_max=y_limits[1])
        y = np.ones((n_samples,)) * y

    # Make x or intervene
    if x is None:
        # Noise distribution
        N_x = _noise_std_zx * norm_samples[2, :] + _noise_offset_zx

        # X = Z * a + N_x
        x = z * a_zx + N_x
    else:
        # Disallow extreme stuff and force
        x = np.clip(x, a_min=x_limits[0], a_max=x_limits[1])
        x = np.ones((n_samples,)) * x

    # Stack data and return
    out = np.stack((x, y), axis=1)
    return out


if __name__ == "__main__":

    _n_tests = 1000  # 1000
    _n_samples = 10000  # 10000

    covs1, covs2, covs3 = [], [], []
    means1, means2, means3 = [], [], []
    for test_nr in tqdm(list(range(_n_tests))):
        c_data_1 = program_1(n_samples=_n_samples, __seed=np.random.randint(0, 100000000))
        c_data_2 = program_2(n_samples=_n_samples, __seed=np.random.randint(0, 100000000))
        c_data_3 = program_3(n_samples=_n_samples, __seed=np.random.randint(0, 100000000))

        covs1.append(np.cov(c_data_1.T))
        covs2.append(np.cov(c_data_2.T))
        covs3.append(np.cov(c_data_3.T))

        means1.append(c_data_1.mean(0))
        means2.append(c_data_2.mean(0))
        means3.append(c_data_3.mean(0))

    covs1 = np.array(covs1)
    covs2 = np.array(covs2)
    covs3 = np.array(covs3)
    means1 = np.array(means1)
    means2 = np.array(means2)
    means3 = np.array(means3)

    with np.printoptions(precision=2, floatmode="fixed"):
        print("\nMean covariance from program 1:")
        print(covs1.mean(0))
        print("\nMean covariance from program 2:")
        print(covs2.mean(0))
        print("\nMean covariance from program 3:")
        print(covs3.mean(0))

        print("\nMean-means:")
        print(means1.mean(0), "   program 1")
        print(means2.mean(0), "   program 2")
        print(means3.mean(0), "   program 3")

