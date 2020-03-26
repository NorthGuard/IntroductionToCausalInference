import shelve
import textwrap

import numpy as np
import pandas as pd

from project.define_server import ExperimentSystem
from project.src.server_util import pandas_print, Storage

if __name__ == "__main__":

    # Connect to shelf and get ids of previously handled emails
    with shelve.open(str(Storage.shelf_path)) as db:

        # Get info
        try:
            prev_ids = db["ids"]  # type: set
            users = db["users"]  # type: dict
            success_ids = db["success_ids"]  # type: set
        except KeyError:
            print("Shelf is empty")
            quit()

    # Header
    print("\n")
    print(textwrap.dedent(f"""
    Server results. \n
    {len(prev_ids)} emails reveived.
    {len(success_ids)} where successfully parsed ({len(success_ids) / len(prev_ids):.1%}).
    """).strip())

    # Extract users data
    names, raw_data = zip(*users.items())

    ##################################################
    # Server table

    # Data
    columns = ["n_emails", "n_samples", "n_experiments", "guesses", "incorrect_guesses"]
    data = np.array([tuple(val.get(k, np.nan) for k in columns) for val in raw_data])

    # Bad emails
    bad = data[:, 0] - data[:, 2] - data[:, 3]
    data = np.concatenate((data, bad[:, None]), axis=1)
    columns.append("bad_emails")

    # Table
    table = pd.DataFrame(
        data=data,
        index=names, columns=columns
    )

    # Users
    print("\n")
    print("-" * 100)
    print("User information")
    with pandas_print():
        print(table)
    print("-" * 100)

    ##################################################
    # Competition

    # Data
    columns = ["n_experiments", "n_samples", "incorrect_guesses"]
    data = np.array([val.get("done", (np.nan,) * 3) for val in raw_data])

    # Compute score
    scores = (
            data[:, 0] * ExperimentSystem.experiment_cost
            + data[:, 1] * ExperimentSystem.sample_cost
            + data[:, 2] * ExperimentSystem.incorrect_guess_cost
    )

    # Append score
    data = np.concatenate((data, scores[:, None]), axis=1)
    columns.append("cost")

    # Make table of users
    table = pd.DataFrame(
        data=data,
        index=names, columns=columns
    )

    # Users
    print("\n")
    print("-" * 100)
    print("Competition table")
    with pandas_print():
        print(table.sort_values("cost"))

    #####################################################
    # Costs

    # Costs
    table = pd.DataFrame(
        data=[[ExperimentSystem.experiment_cost, ExperimentSystem.sample_cost, ExperimentSystem.incorrect_guess_cost]],
        columns=["experiment_cost", "sample_cose", "incorrect_guess_cost"],
        index=["Costs"]

    )

    # Print
    print()
    print("-" * 30)
    with pandas_print():
        print(table)
    print("-" * 100)
    print()
