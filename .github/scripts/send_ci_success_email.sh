#!/usr/bin/env bash

set -euo pipefail

# Build URLs using GitHub expressions

BASE_HTML=$(cat <<EOF
<html>
<body>
  <h2>✅ Github CI - Build ${BRANCH_NAME} SUCCESS</h2>
  <p><strong>Action link:</strong> <a href="${ACTION_URL}">View Action</a></p>
  <p><strong>Branch/PR Name:</strong> ${BRANCH_NAME}</p>
  <p><strong>Correlation Report:</strong> <a href="${REPORT_URL}">View Report</a></p>
  <p><strong>Directory:</strong> ${PWD}</p>
  
  <h3>Correlation Results:</h3>
EOF
)

if [ -f "./util/plotting/correl-html/combined_per_kernel.html" ]; then
  COMBINED_PLOTS=$(cat ./util/plotting/correl-html/combined_per_kernel.html)
  HTML_BODY=$(cat <<EOF
$BASE_HTML
  
  <h3>Combined Correlation Plots:</h3>
  $COMBINED_PLOTS
  </body>
  </html>
EOF
)
else
  HTML_BODY=$(cat <<EOF
$BASE_HTML
  </body>
  </html>
EOF
)
fi

SUBJECT="✅ Github CI - Build ${BRANCH_NAME} SUCCESS"
srun echo "$HTML_BODY" | mail -s "$SUBJECT" -a "Content-Type: text/html; charset=UTF-8" "$GROUP_EMAIL"
echo "Correlation Report at: ${REPORT_URL}."


