"""
This file is modified from the repository:
https://github.com/jdfinch/ezpyzy/blob/main/ezpyzy/send_email.py

Easily send emails programmatically using Python.
"""

import mimetypes
import os
import smtplib
from email.message import EmailMessage

from utils.helpers import get_key


def send_email(
    recipient: str,
    subject: str,
    message: str,
    config_dict: dict | None = None,
    attachments: list[str] | None = None,
):
    """
    Send an email to the specified recipient with the specified subject and message. Create the file:

    ~/.pw/gmail.json or pass in the config_dict parameter with the following format:

    {
        "smtp_server": "smtp.gmail.com",

        "smtp_port": 587,

        "sender_email": "address@gmail.com",

        "sender_password": "app_password"
    }
    """

    if config_dict is None:
        # Load SMTP server settings and credentials from the JSON file
        config_dict = get_key("gmail")

    smtp_server = config_dict["smtp_server"]
    smtp_port = config_dict["smtp_port"]
    sender_email = config_dict["sender_email"]
    sender_password = config_dict["sender_password"]

    # Create a MIMEText object for the message
    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = recipient
    msg["Subject"] = subject

    # Attach the message to the email
    msg.set_content(message)

    if attachments:
        for attachment in attachments:
            # Determine the MIME type of the attachment
            ctype, encoding = mimetypes.guess_type(attachment)
            if ctype is None or encoding is not None:
                ctype = "application/octet-stream"
            mime_type, mime_subtype = ctype.split("/", 1)

            # Read the attachment file
            with open(attachment, "rb") as file:
                attachment_data = file.read()

            # Add the attachment to the email
            msg.add_attachment(
                attachment_data,
                maintype=mime_type,
                subtype=mime_subtype,
                filename=os.path.basename(attachment),
            )

    # Establish a connection to the SMTP server
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()  # Start TLS encryption (optional)
        # Login to your email account
        server.login(sender_email, sender_password)
        # Send the email
        server.sendmail(sender_email, recipient, msg.as_string())
