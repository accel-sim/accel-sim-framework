import os
import smtplib
import getpass
from email.message import EmailMessage

# --- Inputs via environment with sensible defaults ---

username = getpass.getuser()

FROM = f"{username}@ecn.purdue.edu"
TO = os.getenv("GROUP_EMAIL", None)
BRANCH_NAME = os.getenv("BRANCH_NAME", None)
ACTION_URL = os.getenv("ACTION_URL", None)
REPORT_URL = os.getenv("REPORT_URL", None)

if TO is None or BRANCH_NAME is None or ACTION_URL is None or REPORT_URL is None:
    print("Missing required environment variables")
    print(f"TO: {TO}")
    print(f"BRANCH_NAME: {BRANCH_NAME}")
    print(f"ACTION_URL: {ACTION_URL}")
    print(f"REPORT_URL: {REPORT_URL}")
    exit(1)

# --- Build single HTML body ---
combined_path = os.path.join("./util/plotting/correl-html/combined_per_kernel.html")
html_body = f"""
<html>
<body>
  <h2>✅ Github CI - Build {BRANCH_NAME} SUCCESS</h2>
  <p><strong>Action link:</strong> <a href=\"{ACTION_URL}\">View Action</a></p>
  <p><strong>Branch/PR Name:</strong> {BRANCH_NAME}</p>
  <p><strong>Correlation Report:</strong> <a href=\"{REPORT_URL}\">View Report</a></p>
  <h3>Correlation Results Attached.</h3>
  <p><em>The interactive plots are attached as an HTML file.</em></p>
  </body>
  </html>
"""

subject = f"✅ Github CI - Build {BRANCH_NAME} SUCCESS"

# --- Create the Email with HTML alternative ---
msg = EmailMessage()
msg['To'] = TO
msg['Subject'] = subject
msg['From'] = FROM
msg.set_content("This email contains HTML content. If you see this, your client did not render HTML.")
msg.add_alternative(html_body, subtype='html')

# Attach the combined HTML file as well, so recipients can open it in a browser
if os.path.isfile(combined_path):
    try:
        with open(combined_path, 'rb') as f:
            file_data = f.read()
            file_name = os.path.basename(combined_path)
        msg.add_attachment(
            file_data,
            maintype='application',
            subtype='octet-stream',
            filename=file_name,
        )
    except Exception:
        pass

# --- Send the Email ---
with smtplib.SMTP('localhost') as smtp:
    smtp.send_message(msg)

print("Email sent successfully!")