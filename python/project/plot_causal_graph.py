import matplotlib.pyplot as plt

from project.define_server import ExperimentSystem

if __name__ == "__main__":

    # Draw causal system
    causal_system = ExperimentSystem()
    causal_system.draw_causal_graph()

    plt.show(block=True)
