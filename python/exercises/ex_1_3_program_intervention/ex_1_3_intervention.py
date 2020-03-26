import matplotlib.pyplot as plt

from src.ex_1_3_plotting import plot_scatter_w_histograms
from src.ex_1_3_programs import program_1, program_2, program_3

# Settings
n_samples = 40  # 40
x_intervention = None  # None
y_intervention = None  # None
fit_normal_distribution = False  # False

#######

marker_size = 70

# Sample programs
data1 = program_1(n_samples=n_samples, x=x_intervention, y=y_intervention)
data2 = program_2(n_samples=n_samples, x=x_intervention, y=y_intervention)
data3 = program_3(n_samples=n_samples, x=x_intervention, y=y_intervention)

# Compute limits
x_lim = (min(data1[:, 0].min(), data2[:, 0].min(), data3[:, 0].min()) * 1.15,
         max(data1[:, 0].max(), data2[:, 0].max(), data3[:, 0].max()) * 1.15) if x_intervention is None else None
y_lim = (min(data1[:, 1].min(), data2[:, 1].min(), data3[:, 1].min()) * 1.15,
         max(data1[:, 1].max(), data2[:, 1].max(), data3[:, 1].max()) * 1.15) if y_intervention is None else None

fig_size = (6, 6)

fig = plt.figure(1)
fig.clear()
fig.set_size_inches(*fig_size)
plot_scatter_w_histograms(
    data=data1, color_nr=0, marker="^", title="Program 1", infer_gaussian=fit_normal_distribution,
    x_lim=x_lim, y_lim=y_lim, marker_size=marker_size,
)

fig = plt.figure(2)
fig.clear()
fig.set_size_inches(*fig_size)
plot_scatter_w_histograms(
    data=data2, color_nr=1, marker=">", title="Program 2", infer_gaussian=fit_normal_distribution,
    x_lim=x_lim, y_lim=y_lim, marker_size=marker_size,
)

fig = plt.figure(3)
fig.clear()
fig.set_size_inches(*fig_size)
plot_scatter_w_histograms(
    data=data3, color_nr=2, marker="<", title="Program 3", infer_gaussian=fit_normal_distribution,
    x_lim=x_lim, y_lim=y_lim, marker_size=marker_size,
)

plt.show()
