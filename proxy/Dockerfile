FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["mitmdump", "-s", "proxy.py", "--listen-host", "0.0.0.0", "--listen-port", "8080", "--set", "confdir=/certs", "--set", "block_global=false", "--ignore-hosts", "(^|\\.)apple\\.com$|(^|\\.)icloud\\.com$|(^|\\.)mzstatic\\.com$|(^|\\.)google\\.com$|(^|\\.)googleapis\\.com$|(^|\\.)gstatic\\.com$|(^|\\.)googlevideo\\.com$"]
