import os
import sys
import smtplib
import getpass
import argparse
from email.message import EmailMessage

"""Command-line interface

New argument-based interface (no positional args):
  --email-type {success,failure}
  --urgent (flag, only for failure emails)
"""

parser = argparse.ArgumentParser(description="Send CI result email (success/failure)")
parser.add_argument("-t", "--email-type", required=True, choices=["success", "failure"], help="Type of email to send")
parser.add_argument("-u", "--urgent", action="store_true", help="Mark failure email as urgent (uses ❗). If not set, treated as warning (⚠️)")

args = parser.parse_args()

EMAIL_TYPE = args.email_type
URGENCY = "urgent" if args.urgent else "warning"

username = getpass.getuser()

FROM = f"{username}@ecn.purdue.edu"
TO = os.getenv("GROUP_EMAIL", None)
BRANCH_NAME = os.getenv("BRANCH_NAME", None)
ACTION_URL = os.getenv("ACTION_URL", None)
REPORT_URL = os.getenv("REPORT_URL", None)
# No failed-jobs enumeration used anymore

if TO is None or BRANCH_NAME is None or ACTION_URL is None:
    print("Missing required environment variables")
    print(f"TO: {TO}")
    print(f"BRANCH_NAME: {BRANCH_NAME}")
    print(f"ACTION_URL: {ACTION_URL}")
    exit(1)

# --- Build HTML body based on email type ---
if EMAIL_TYPE == "success":

    # Build correlation plot links if REPORT_URL provided
    plots_html = ""
    if REPORT_URL:
        # CORREL_GPUS: "<dir>:Label" pairs, the dirs under REPORT_URL holding combined_per_*.html
        if not os.getenv("CORREL_GPUS"):
            sys.exit("REPORT_URL is set but CORREL_GPUS is not")
        gpus = os.getenv("CORREL_GPUS").split()
        links = ""
        for gpu in gpus:
            prefix, label = gpu.split(":")
            for kind, name in (("kernel", "Per Kernel"), ("app", "Per App")):
                url = os.path.join(REPORT_URL, prefix, f"combined_per_{kind}.html")
                links += f'    <li><a href="{url}">{label} - {name}</a></li>\n'
        plots_html = f"""
  <h3>Correlation Plots</h3>
  <ul>
{links}  </ul>
"""
    html_body = f"""
<html>
<body>
  <h2>✅ Github CI - Build {BRANCH_NAME} SUCCESS</h2>
  <p><strong>Action link:</strong> <a href=\"{ACTION_URL}\">View Action</a></p>
  <p><strong>Branch/PR Name:</strong> {BRANCH_NAME}</p>
{plots_html}
  </body>
  </html>
"""
    subject = f"✅ Github CI - Build {BRANCH_NAME} SUCCESS"
else:  # failure
    # Choose emoji based on urgency
    emoji = "❗" if URGENCY == 'urgent' else "⚠️"

    html_body = f"""
<html>
<body>
  <h2>{emoji} Github CI - Build {BRANCH_NAME} FAILED</h2>
  <p><strong>Action link:</strong> <a href=\"{ACTION_URL}\">View Action</a></p>
  <p><strong>Branch/PR Name:</strong> {BRANCH_NAME}</p>
  <p style=\"color: red;\"><strong>Please check the action logs for details.</strong></p>
  </body>
  </html>
"""
    subject = f"{emoji}Github CI FAILED - Build {BRANCH_NAME}"

# --- Create the Email with HTML alternative ---
msg = EmailMessage()
msg['To'] = TO
msg['Subject'] = subject
msg['From'] = FROM
msg.set_content("This email contains HTML content. If you see this, your client did not render HTML.")
msg.add_alternative(html_body, subtype='html')

# --- Send the Email ---
with smtplib.SMTP('localhost') as smtp:
    smtp.send_message(msg)

print(f"{EMAIL_TYPE.title()} email sent successfully!")