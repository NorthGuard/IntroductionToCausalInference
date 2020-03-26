import numpy as np

from project.src.causal_system import CausalSystem


class ServerSettings:

    # Email information  -> you need to turn on "Less secure app access" on Gmail
    username = "intervention.experiment@gmail.com"  # <*\label{code:email}*>
    server_password = "hnJk9FYAWN03Er2p"  # <*\label{code:password}*>

    # Email provider information
    imap_host = "imap.gmail.com"  # <*\label{code:imap_host}*>
    smtp_host = "smtp.gmail.com"  # <*\label{code:smtp_host}*>
    smtp_port = 587  # <*\label{code:smtp_port}*>

    # Other settings
    check_email_delay = 2  # <*\label{code:server_check_time}*>

    # Emails with access to experiment   # <*\label{code:allowed_emails_start}*>
    allowed_emails = """
    intervention.experiment@gmail.com
    jepno@dtu.dk
    student_email_1@university.com
    student_email_2@university.com
    """  # <*\label{code:allowed_emails_stop}*>


# Define the causal system
class ExperimentSystem(CausalSystem):  # <*\label{code:ExperimentSystem}*>
    _project_password = "Open_Sesame"  # <*\label{code:project_password}*>

    # Score settings for competition <*\label{code:competition}*>
    experiment_cost = 20
    sample_cost = 1
    incorrect_guess_cost = 70

    # Causal model
    def _sample(self, n_samples):  # <*\label{code:_sample}*>
        # Using predefined distributions
        self["X"] = self.normal(mu=2, std=3)    # <*\label{code:predef_distribution_1}*>
        self["Y"] = self.binary(p_success=0.8)    # <*\label{code:predef_distribution_2}*>
        self["Z"] = self.categorical([7, 2, 1])    # <*\label{code:predef_distribution_3}*>

        # You can also use numpy sampling
        #  - but you have to do it correctly by using _n_samples
        self["F"] = np.random.beta(a=5, b=3, size=n_samples)  # <*\label{code:numpy_sample}*>

        # Combining is fine
        self["G"] = self.normal(mu=0, std=1) * self["X"] + self["F"]  # <*\label{code:combine}*>

        # This variable is hidden (because of the "_")
        # It will not be returned in the data unless you have the password
        self["_H"] = self.normal(mu=0, std=1)  # <*\label{code:hidden}*>

        # This variable has a hidden confounder
        self["I"] = self["F"] + self["_H"] + self.normal(mu=0, std=0.2)  # <*\label{code:variable_with_confounder}*>

        # Ordering without hint of causal structure
        self._ordering = ['I', 'Z', 'F', '_H', 'X', 'G', 'Y']  # <*\label{code:ordering}*>
