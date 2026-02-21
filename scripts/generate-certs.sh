#!/bin/bash

# Create certificates directory if it doesn't exist
mkdir -p certificates

# Generate CA private key and certificate
openssl genrsa -des3 -out certificates/rootCA.key 2048
openssl req -x509 -new -nodes -key certificates/rootCA.key -sha256 -days 1024 -out certificates/rootCA.pem \
  -subj "/C=TR/ST=Istanbul/L=Istanbul/O=Sardag Software/OU=Development/CN=Sardag Local CA"

# Generate server private key
openssl genrsa -out certificates/localhost.key 2048

# Create CSR configuration
cat > certificates/localhost.conf << EOF
[req]
default_bits = 2048
prompt = no
default_md = sha256
req_extensions = req_ext
distinguished_name = dn

[dn]
C=TR
ST=Istanbul
L=Istanbul
O=Sardag Software
OU=Development
CN=localhost

[req_ext]
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = *.localhost
IP.1 = 127.0.0.1
EOF

# Generate CSR
openssl req -new -key certificates/localhost.key -out certificates/localhost.csr -config certificates/localhost.conf

# Generate certificate
openssl x509 -req -in certificates/localhost.csr -CA certificates/rootCA.pem -CAkey certificates/rootCA.key \
  -CAcreateserial -out certificates/localhost.crt -days 365 -sha256 -extensions req_ext -extfile certificates/localhost.conf

# Clean up
rm certificates/localhost.csr certificates/localhost.conf

echo "SSL certificates generated successfully!"
echo "Add certificates/rootCA.pem to your system's trusted certificates to avoid browser warnings."