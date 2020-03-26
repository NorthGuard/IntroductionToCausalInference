import re
import shelve
import textwrap
from ast import literal_eval
from datetime import datetime
from pathlib import Path
from time import sleep

from imapclient import IMAPClient, exceptions
from project.define_server import ServerSettings, ExperimentSystem
from project.src.server_util import Storage, slugify, pandas_print, send_mail


class Server:
    def __init__(self):
        self.answer_emails = True

        # Handle emails with access
        self.allowed_emails = {val.strip() for val in ServerSettings.allowed_emails.split("\n") if val.strip()}

        # Make causal system
        self.causal_system = ExperimentSystem()

        # Connect to shelf and get ids of previously handled emails
        with shelve.open(str(Storage.shelf_path)) as db:
            if "ids" in db:
                self.prev_ids = db["ids"]
                self.success_ids = db["success_ids"]
                self.emails = db["emails"]
                self.users = db["users"]
            else:
                self.prev_ids = set()
                self.success_ids = set()
                self.emails = dict()
                self.users = dict()
            db["allowed_emails"] = self.allowed_emails

    def print(self, *args, **kwargs):
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), end=" -> ")
        print(*args, **kwargs)

    def get_emails(self):
        self.print("Checking emails")

        # Log into mail
        response = None
        messages = None
        try:
            with IMAPClient(host=ServerSettings.imap_host) as client:
                client.login(username=ServerSettings.username, password=ServerSettings.server_password)
                client.select_folder("inbox")

                # Get messages
                messages = client.search(['NOT', 'DELETED'])

                # Filter out previous emails
                messages = [val for val in messages if val not in self.prev_ids]

                # Fetch envelopes
                response = client.fetch(messages=messages, data='ENVELOPE')
        except exceptions.IMAPClientError:
            pass

        return response, messages

    @staticmethod
    def parse_envelope(envelope):
        # Get subject
        try:
            subject = envelope.subject.decode()
        except AttributeError:
            subject = ""

        # Get sender
        sender = envelope.sender[0]
        sender = sender.mailbox.decode() + "@" + sender.host.decode()

        # Return
        return subject, sender

    def handle_allowed_email(self, subject: str, sender: str, message_id):
        error_message = None
        samples = None
        graph_is_correct = None
        ran_experiment = False

        # Catch most errors due to bad email
        try:
            # Check for guess of causal graph
            search = re.search("^\\s*guess:(.*)", subject.lower())
            if search:
                error_message = "Could not check graph for correctness"

                # Get graph-guess very precisely
                guess = re.search("^[^{}()\\[\\]]*([\\[({].*[\\]})])[^{}()\\[\\]]*$", subject).group(1)

                # Un-escape strings
                guess = guess.replace(r"\'", "'").replace(r'\"', '"')

                # Ensure causal system has been sampled
                _ = self.causal_system.sample(1)

                # Check correctness
                graph_is_correct = self.causal_system.check_correct_graph(edge_list=guess)

                # Information for email
                if graph_is_correct:
                    subject_line = "CORRECT GRAPH!"
                    text = f"The following graph is CORRECT: \n{guess}"
                else:
                    subject_line = "Incorrect graph."
                    text = f"The following graph is INCORRECT: \n{guess}"

                # Send email
                if self.answer_emails:
                    error_message = "Could not send email with guess-answer"
                    send_mail(
                        send_from=ServerSettings.username,
                        send_to=sender,
                        subject=subject_line,
                        text=text,
                        username=ServerSettings.username,
                        password=ServerSettings.server_password,
                        email_smtp_server=ServerSettings.smtp_host,
                        email_smtp_port=ServerSettings.smtp_port,
                    )

            # User wants samples
            else:

                # Parse subject line
                error_message = "Cannot parse subject line"
                search = re.search("^(\\d+)[ ,\\s]*([^\n]*)", subject)
                n_samples = int(search.group(1))
                settings_str = str(search.group(2)).strip()

                # Evaluate settings
                error_message = "Cannot extract settings"
                settings = dict()
                settings_parts = re.findall("([_\\w]+)=([\\d.\\w]+)", settings_str)
                for key, value in settings_parts:
                    if key == "password":
                        settings[key] = str(value)
                    else:
                        settings[key] = literal_eval(value)

                # Make samples
                error_message = "Could not make samples"
                samples = self.causal_system.sample(n_samples, **settings)

                #####
                # Send response

                if self.answer_emails:

                    error_message = "Path problems"

                    # Directory for specific user
                    user_directory = Path(Storage.data_path, slugify(sender.replace("@", "_at_")))
                    user_directory.mkdir(parents=True, exist_ok=True)

                    # File path
                    file_path = Path(user_directory, f"data_{message_id}.csv")
                    file_path_readable = Path(user_directory, f"data_{message_id}_readable.txt")
                    assert not file_path.exists()
                    assert not file_path_readable.exists()

                    # Make data-file
                    with file_path.open("w") as file:
                        samples.to_csv(
                            path_or_buf=file, sep=",", header=True, index=True,
                        )

                    # Make human readable data-file
                    with pandas_print():
                        with file_path_readable.open("w") as file:
                            file.write(samples.__repr__())

                    # Send email
                    error_message = "Could not send email with samples"
                    line = f"Data for query: {subject}"
                    send_mail(
                        send_from=ServerSettings.username,
                        send_to=sender,
                        subject=line,
                        text=line,
                        username=ServerSettings.username,
                        password=ServerSettings.server_password,
                        files=[file_path, file_path_readable],
                        email_smtp_server=ServerSettings.smtp_host,
                        email_smtp_port=ServerSettings.smtp_port,
                    )

                # This was an experiment
                ran_experiment = True

            # Success
            error_message = "SUCCESS"
            email_is_success = True
            self.print(f"\t\tEmail {message_id}: {error_message}, [{subject}], [{sender}]")

        # Bad email
        except (ValueError, AttributeError):
            email_is_success = False
            if sender is not None:
                self.print(f"\t\tEmail {message_id}: {error_message}, [{subject}], {'[' + str(sender) + ']':50s}<--")
            else:
                self.print(f"\t\tEmail {message_id}: {error_message} {'[-]':50s}<--")

            # Send answer and this time ignore all errors
            if self.answer_emails:
                try:
                    subject_line = f"Unknown query: {subject}"
                    text = textwrap.dedent("""
                    Example of experiment query: 
                        20, X=1
                    
                    Example of guess query:
                        guess: [('A', 'B'), ('B', 'C')]
                    
                    """)
                    send_mail(
                        send_from=ServerSettings.username,
                        send_to=sender,
                        subject=subject_line,
                        text=text,
                        username=ServerSettings.username,
                        password=ServerSettings.server_password,
                        email_smtp_server=ServerSettings.smtp_host,
                        email_smtp_port=ServerSettings.smtp_port,
                    )
                except (ValueError, AttributeError):
                    pass

        # Update storage
        self.update_persistent_memory(
            message_id=message_id,
            email_is_success=email_is_success,
            error_message=error_message,
            subject=subject,
            sender=sender,
            samples=samples,
            graph_is_correct=graph_is_correct,
            ran_experiment=ran_experiment,
        )

        # Return
        return email_is_success, error_message

    def update_persistent_memory(self, message_id, email_is_success, error_message, subject, sender,
                                 samples, graph_is_correct, ran_experiment):
        # Handle email-id
        self.prev_ids.add(message_id)

        # Handle success
        if email_is_success:
            self.success_ids.add(message_id)

        # Handle email data
        self.emails[message_id] = dict(
            message=error_message,
            subject=subject,
            sender=sender,
            success=email_is_success,
        )

        # Handle user emails
        user_info = self.users.get(sender, dict())
        user_info["n_emails"] = user_info.get("n_emails", 0) + 1

        # Handle user experiments
        user_info["n_experiments"] = user_info.get("n_experiments", 0) + ran_experiment

        # Handle user samples
        if samples is not None:
            user_info["n_samples"] = user_info.get("n_samples", 0) + samples.shape[0]
        self.users[sender] = user_info

        # Handle user guesses
        if graph_is_correct is not None:
            user_info["guesses"] = user_info.get("guesses", 0) + 1
            user_info["incorrect_guesses"] = user_info.get("incorrect_guesses", 0) + (not graph_is_correct)

            # Check for correct guess - finish student
            if graph_is_correct and "done" not in user_info:
                user_info["done"] = (user_info["n_experiments"], user_info["n_samples"], user_info["incorrect_guesses"])

        # Store persistently
        with shelve.open(str(Storage.shelf_path)) as db:
            db["ids"] = self.prev_ids
            db["emails"] = self.emails
            db["users"] = self.users
            db["success_ids"] = self.success_ids

    def __call__(self, answer_emails=True, single_run=False):
        self.answer_emails = answer_emails

        # Keep reading emails
        while True:
            print("")

            # Get emails
            response, messages = self.get_emails()

            # Didn't success in connecting to email
            if response is None:
                self.print("\t\tCan not connect to email provider!")

            # No new emails
            elif not messages:
                self.print("\t\tNo new emails.")

            else:

                # Go through messages
                for message_id, message_data in response.items():

                    # Parse envelope
                    subject, sender = self.parse_envelope(envelope=message_data[b"ENVELOPE"])

                    # Check sender
                    if sender in self.allowed_emails:

                        # Handle email
                        self.handle_allowed_email(subject=subject, sender=sender, message_id=message_id)

                    else:
                        error_message = "Email address not allowed access."
                        self.print(f"\t\tEmail {message_id}: {error_message} {'[-]':50s}<--")

            #########################################

            # Sleep a bit
            if single_run:
                break
            self.print(f"Sleeping {ServerSettings.check_email_delay}s")
            sleep(ServerSettings.check_email_delay)
