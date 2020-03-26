import os
import re
import smtplib
import unicodedata
from pathlib import Path

import pandas as pd
from os.path import basename
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate


def send_mail(
        send_from, send_to, subject, text,
        username, password,
        files=None,
        email_smtp_server="smtp.gmail.com", email_smtp_port=587):
    if isinstance(send_to, str):
        send_to = [send_to]
    assert isinstance(send_to, list)

    # Make message
    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = COMMASPACE.join(send_to)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    # Do text
    msg.attach(MIMEText(text))

    # Attach files
    for f in files or []:
        with open(f, "rb") as fil:
            part = MIMEApplication(
                fil.read(),
                Name=basename(f)
            )
        # After the file is closed
        part['Content-Disposition'] = 'attachment; filename="%s"' % basename(f)
        msg.attach(part)

    # Login
    server = smtplib.SMTP(email_smtp_server, email_smtp_port)
    server.ehlo()
    server.starttls()
    server.login(username, password)

    # Send email
    server.sendmail(send_from, send_to, msg.as_string())

    # Log out
    server.quit()


def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    value = value.lower().replace(".", "_")
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = re.sub("[^\\w\\s\\d-]", "", value).strip()
    value = re.sub("[\\s]+", "_", value)
    return value


def pandas_print():
    return pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 300)


class Storage:
    # Working directory
    _cwd = Path.cwd()
    if _cwd.name == "python":
        os.chdir("project")
        _cwd = Path.cwd()
    assert Path.cwd().name == "project"

    # Main path
    main = Path(Path.cwd(), "storage")

    # Prepare path for internal storage shelf
    shelf_path = Path(main, "_storage", "previous_emails")
    shelf_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare path for student data
    data_path = Path(main, "student_data")
    data_path.parent.mkdir(parents=True, exist_ok=True)