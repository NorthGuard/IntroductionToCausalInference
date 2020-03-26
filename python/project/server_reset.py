import shutil
from project.src.server_util import Storage


def reset_server(complete=True):
    if Storage.shelf_path.parent.exists():
        for path in Storage.shelf_path.parent.glob("*"):
            path.unlink()

    if complete:
        if Storage.data_path.exists():
            for path in Storage.data_path.glob("*"):
                shutil.rmtree(path)

    # Remake structure
    Storage.data_path.mkdir(parents=True, exist_ok=True)
    Storage.shelf_path.parent.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":

    # Ask first
    answer = input("Do you want to reset the server's memory? [y/N] ")

    # Reset server storage
    if "y" in answer.lower():
        reset_server()
        print("Server has been reset.")
    else:
        print("Exiting.")
