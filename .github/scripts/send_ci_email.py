import os
import sys
import smtplib
import getpass
from email.message import EmailMessage

# --- Inputs via command line arguments and environment ---

# Parse command line arguments
if len(sys.argv) < 2:
    print("Usage: python3 send_ci_email.py <email_type> [urgency]")
    print("  email_type: 'success' or 'failure'")
    print("  urgency: 'urgent' or 'warning' (optional, defaults to 'urgent' for failures)")
    sys.exit(1)

EMAIL_TYPE = sys.argv[1].lower()
if EMAIL_TYPE not in ['success', 'failure']:
    print("Error: email_type must be 'success' or 'failure'")
    sys.exit(1)

# Get urgency level (second argument, optional)
URGENCY = sys.argv[2].lower() if len(sys.argv) > 2 else ('urgent' if EMAIL_TYPE == 'failure' else None)
if URGENCY and URGENCY not in ['urgent', 'warning']:
    print("Error: urgency must be 'urgent' or 'warning'")
    sys.exit(1)

username = getpass.getuser()

FROM = f"{username}@ecn.purdue.edu"
TO = os.getenv("GROUP_EMAIL", None)
BRANCH_NAME = os.getenv("BRANCH_NAME", None)
ACTION_URL = os.getenv("ACTION_URL", None)
REPORT_URL = os.getenv("REPORT_URL", None)
FAILED_JOBS = os.getenv("FAILED_JOBS", None)

if TO is None or BRANCH_NAME is None or ACTION_URL is None:
    print("Missing required environment variables")
    print(f"TO: {TO}")
    print(f"BRANCH_NAME: {BRANCH_NAME}")
    print(f"ACTION_URL: {ACTION_URL}")
    exit(1)

# REPORT_URL is only required for success emails
if EMAIL_TYPE == "success" and REPORT_URL is None:
    print("Missing required environment variable for success email")
    print(f"REPORT_URL: {REPORT_URL}")
    exit(1)

# --- Build HTML body based on email type ---
if EMAIL_TYPE == "success":
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
else:  # failure
    combined_path = None  # No file to attach for failures
    
    # Choose emoji based on urgency
    emoji = "❗" if URGENCY == 'urgent' else "⚠️"
    
    # Build failed jobs information
    failed_jobs_info = ""
    if FAILED_JOBS:
        failed_jobs_list = FAILED_JOBS.split(',') if FAILED_JOBS else []
        if failed_jobs_list:
            failed_jobs_info = f"""
  <p><strong>Failed Jobs:</strong></p>
  <ul>
"""
            for job in failed_jobs_list:
                if job.strip():  # Skip empty strings
                    failed_jobs_info += f"    <li style=\"color: red;\">{job.strip()}</li>\n"
            failed_jobs_info += "  </ul>"
    
    html_body = f"""
<html>
<body>
  <h2>{emoji} Github CI - Build {BRANCH_NAME} FAILED</h2>
  <p><strong>Action link:</strong> <a href=\"{ACTION_URL}\">View Action</a></p>
  <p><strong>Branch/PR Name:</strong> {BRANCH_NAME}</p>
{failed_jobs_info}
  <p style="color: red;"><strong>Please check the action logs for details.</strong></p>
  </body>
  </html>
"""
    # Build subject line with failed jobs if available
    if FAILED_JOBS and FAILED_JOBS.strip():
        failed_jobs_list = [job.strip() for job in FAILED_JOBS.split(',') if job.strip()]
        if failed_jobs_list:
            jobs_text = ', '.join(failed_jobs_list[:2])  # Show first 2 jobs
            if len(failed_jobs_list) > 2:
                jobs_text += f" (+{len(failed_jobs_list)-2} more)"
            subject = f"{emoji}Github CI FAILED - {jobs_text} - {BRANCH_NAME}"
        else:
            subject = f"{emoji}Github CI FAILED - Build {BRANCH_NAME}"
    else:
        subject = f"{emoji}Github CI FAILED - Build {BRANCH_NAME}"

# --- Create the Email with HTML alternative ---
msg = EmailMessage()
msg['To'] = TO
msg['Subject'] = subject
msg['From'] = FROM
msg.set_content("This email contains HTML content. If you see this, your client did not render HTML.")
msg.add_alternative(html_body, subtype='html')

# Attach the combined HTML file for success emails only
if EMAIL_TYPE == "success" and combined_path and os.path.isfile(combined_path):
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

print(f"{EMAIL_TYPE.title()} email sent successfully!")