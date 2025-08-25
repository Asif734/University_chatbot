# app/utils/email_alert.py
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from app.config import EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASSWORD, ADMIN_EMAIL


def send_alert_if_needed(user, user_message):
    """
    Send an email alert to the admin if a mental health risk is detected.
    """
    try:
        # Create the email
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = ADMIN_EMAIL
        msg['Subject'] = f"[ALERT] Potential Mental Health Risk: {user.name} ({user.id})"

        body = f"""
        The following message from a student triggered a mental health risk alert:

        Student: {user.name} (ID: {user.id})
        Message: {user_message}

        Please review and take appropriate action.
        """

        msg.attach(MIMEText(body, 'plain'))

        # Send the email
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()

        print(f"Alert email sent for student {user.id}")

    except Exception as e:
        print(f"Failed to send alert email: {str(e)}")
