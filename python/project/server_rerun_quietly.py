from project.server_reset import reset_server
from project.src.server_machinery import Server

if __name__ == "__main__":
    # Reset only internal storage
    reset_server(complete=False)

    # Make server
    server = Server()

    # Run
    server(answer_emails=False, single_run=True)
