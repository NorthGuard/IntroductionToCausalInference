import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde

from src.ex_1_3_programs import x_limits, y_limits


def plot_scatter_w_histograms(data, ax=None, color_nr=0, hist_ratio=0.1, do_density=True, infer_gaussian=True,
                              marker_size=50, marker="^", title=None, x_lim=None, y_lim=None):
    x_lim = x_limits if x_lim is None else x_lim
    y_lim = y_limits if y_lim is None else y_lim

    data = np.array(data)
    fontsize = 17
    small_factor = 0.6
    big_factor = 1.2

    # Ensure axes
    offset = 0.12
    ax = ax if ax is not None else plt.subplot(position=[offset, offset, 1 - offset * 1.2, 1 - offset * 1.2])

    # Make grid
    subgrid = GridSpecFromSubplotSpec(
        nrows=2, ncols=2, subplot_spec=ax,
        wspace=0.000, hspace=0.000, height_ratios=[hist_ratio, 1 - hist_ratio],
        width_ratios=[1 - hist_ratio, hist_ratio]
    )

    # Settings
    colors = sns.color_palette("dark")
    fade_colors = sns.color_palette("muted")
    facecolor = fade_colors[color_nr]
    line_color = colors[color_nr]
    marker_color = colors[color_nr]
    contour_color = fade_colors[color_nr]

    # Scatterplot
    sc_ax = plt.subplot(subgrid[1, 0])
    sc_ax.scatter(data[:, 0], data[:, 1], color=marker_color, s=marker_size, marker=marker)
    sc_ax.set_xlabel("x", fontsize=int(fontsize * big_factor), fontweight="bold")
    sc_ax.set_ylabel("y", fontsize=int(fontsize * big_factor), fontweight="bold")
    sc_ax.spines["right"].set_visible(False)
    sc_ax.spines["top"].set_visible(False)
    sc_ax.set_xlim(*x_lim)
    sc_ax.set_ylim(*y_lim)

    # Tick-sizes
    for tick in sc_ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=int(fontsize * small_factor))
    for tick in sc_ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize=int(fontsize * small_factor))

        # Top histogram
    th_ax = plt.subplot(subgrid[0, 0])
    th_ax.tick_params(axis="x", labelbottom=False)
    th_ax.set_xlim(*x_lim)
    c_data = data[:, 0]
    if do_density:
        try:
            density = gaussian_kde(c_data)
        except np.linalg.LinAlgError:
            temp = c_data + np.random.randn(*c_data.shape) * 0.01
            density = gaussian_kde(temp)
        x = np.linspace(*x_lim, 200)
        # noinspection PyArgumentList
        y = density(x)
        # ax.hist(data[:, 0], bins=50, density=True)
        th_ax.plot(x, y, color=line_color)
        th_ax.plot([x.min(), x.max()], [0, 0], color="k", linewidth=2)
        th_ax.fill_between(x, y, facecolors=facecolor)
    else:
        th_ax.hist(c_data, bins=50, density=True, facecolor=facecolor, edgecolor=line_color, range=x_lim)
    c_y_lim = th_ax.get_ylim()
    th_ax.set_ylim(0, c_y_lim[1])
    th_ax.spines["right"].set_visible(False)
    th_ax.spines["top"].set_visible(False)
    th_ax.spines["left"].set_visible(False)
    th_ax.spines["bottom"].set_visible(False)
    th_ax.set_xticks([])
    th_ax.set_yticks([])
    th_ax.set_xlim(*x_lim)

    # Right histogram
    rh_ax = plt.subplot(subgrid[1, 1])
    rh_ax.tick_params(axis="y", labelleft=False)
    rh_ax.set_ylim(*y_lim)
    c_data = data[:, 1]
    if do_density:
        try:
            density = gaussian_kde(c_data)
        except np.linalg.LinAlgError:
            temp = c_data + np.random.randn(*c_data.shape) * 0.01
            density = gaussian_kde(temp)
        x = np.linspace(*y_lim, 200)
        # noinspection PyArgumentList
        y = density(x)
        # ax.hist(data[:, 1], bins=50, density=True, orientation='horizontal')
        rh_ax.plot(y, x, color=line_color)
        rh_ax.plot([0, 0], [x.min(), x.max()], color="k", linewidth=2)
        rh_ax.fill_betweenx(y=x, x1=y, facecolors=facecolor)
    else:
        rh_ax.hist(c_data, bins=50, density=True, orientation="horizontal", facecolor=facecolor, edgecolor=line_color,
                   range=y_lim)
    c_x_lim = rh_ax.get_xlim()
    rh_ax.set_xlim(0, c_x_lim[1])
    rh_ax.spines["right"].set_visible(False)
    rh_ax.spines["top"].set_visible(False)
    rh_ax.spines["left"].set_visible(False)
    rh_ax.spines["bottom"].set_visible(False)
    rh_ax.set_xticks([])
    rh_ax.set_yticks([])

    # Ranges of plot
    x_range = max(sc_ax.get_xlim()) - min(sc_ax.get_xlim())
    y_range = max(sc_ax.get_ylim()) - min(sc_ax.get_ylim())

    # Title
    y_start = max(sc_ax.get_ylim()) - y_range * 0.02
    x_start = min(sc_ax.get_xlim()) + x_range * 0.02
    if title is not None:
        sc_ax.text(
            x=x_start, y=y_start,
            ha="left", va="top",
            s=title, fontsize=int(fontsize * big_factor), fontweight="bold"
        )
        y_start -= y_range * 0.08

    if infer_gaussian:

        # Compute mean
        mu = data.mean(0)

        # Compute covariance
        cov = np.cov(data[:, 0] - mu[0], data[:, 1] - mu[1])

        # Eigenvectors
        eigen_values, eigen_vectors = np.linalg.eig(cov)
        loc = np.argsort(eigen_values)
        largest_vector = eigen_vectors[:, loc[1]]

        # Angle of ellipse
        angle = np.arctan(largest_vector[1] / largest_vector[0])
        angle = (angle + np.pi) if angle < 0 else angle

        # Contour lines
        for factor in [1, 2, 3]:

            # Ellipse width and height
            width = (cov.diagonal().max()) * factor
            height = (cov.diagonal().min()) * factor

            # Ellipse
            ellipse = Ellipse(xy=mu, width=width, height=height, angle=angle * 180 / np.pi, fill=False,
                              edgecolor=contour_color, linewidth=2)
            sc_ax.add_artist(ellipse)

        # Add information
        text_str = (
            f"Estimated mean: [{mu[0]:.2f}, {mu[1]:.2f}] \n" +
            f"Estimated covariance: \n" +
            f"            [ {cov[0, 0]:5.2f}, {cov[0, 1]:5.2f} ] \n" +
            f"            [ {cov[1, 0]:5.2f}, {cov[1, 1]:5.2f} ] \n"
        )

        # Write out parameters
        sc_ax.text(
            x=x_start, y=y_start,
            ha="left", va="top",
            s=text_str, fontsize=int(fontsize * small_factor)
        )
