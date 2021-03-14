import pycurl
import subprocess
from urllib.parse import urlencode
from io import BytesIO
import certifi

crl = pycurl.Curl()
# import google.auth
# import google.auth.transport.requests
# creds, project = google.auth.default()
token = subprocess.getoutput("gcloud auth application-default print-access-token")
print(token)
# auth_req = google.auth.transport.requests.Request()
# creds.refresh(auth_req)
# print(auth_req)
# creds.refresh(auth_req)

# now we can print the access token
# print(creds.token)
buffer = BytesIO()
crl.setopt(pycurl.CAINFO, certifi.where())

crl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer '+token])
crl.setopt(crl.URL, 'https://healthcare.googleapis.com/v1beta1/projects/quexleval/locations/us-central1/services/nlp:analyzeEntities'.encode('utf-8').strip())
data = {'nlpService': 'projects/quexleval/locations/us-central1/services/nlp',
        'documentContent': 'I want to tell you about a novel drug against measles and chickenpox. It is based on antibody XYZab and is formulated as a pill. Side effects of allergies are rare. The medicine is effective in 95% of patients according to clinical trial CT-1234.'}
pf = urlencode(data)
crl.setopt(crl.POSTFIELDS, pf)
crl.setopt(crl.WRITEDATA, buffer)

crl.perform()
print(buffer.getvalue())
crl.close()
# pf = urlencode(data)