package main

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/elazarl/goproxy"
	"github.com/pkg/errors"
)

func loadOrCreateCA(certFile, keyFile string) (tls.Certificate, error) {
	_, certErr := os.Stat(certFile)
	_, keyErr := os.Stat(keyFile)

	if os.IsNotExist(certErr) || os.IsNotExist(keyErr) {
		if err := generateCA(certFile, keyFile); err != nil {
			return tls.Certificate{}, errors.Wrap(err, "generate CA")
		}
	}

	cert, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		return tls.Certificate{}, errors.Wrap(err, "load CA")
	}

	// Parse the leaf so goproxy can inspect the cert fields.
	cert.Leaf, err = x509.ParseCertificate(cert.Certificate[0])
	if err != nil {
		return tls.Certificate{}, errors.Wrap(err, "parse CA leaf")
	}

	return cert, nil
}

func generateCA(certFile, keyFile string) error {
	if err := os.MkdirAll(filepath.Dir(certFile), 0o755); err != nil {
		return errors.Wrap(err, "create certs dir")
	}

	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return errors.Wrap(err, "generate key")
	}

	serial, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	if err != nil {
		return errors.Wrap(err, "generate serial")
	}

	tmpl := &x509.Certificate{
		SerialNumber: serial,
		Subject: pkix.Name{
			CommonName:   "Salus Proxy CA",
			Organization: []string{"Salus"},
		},
		NotBefore:             time.Now().Add(-time.Minute),
		NotAfter:              time.Now().Add(10 * 365 * 24 * time.Hour),
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
		BasicConstraintsValid: true,
		IsCA:                  true,
	}

	certDER, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		return errors.Wrap(err, "create cert")
	}

	cf, err := os.OpenFile(certFile, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return errors.Wrap(err, "open cert file")
	}
	defer cf.Close()
	if err := pem.Encode(cf, &pem.Block{Type: "CERTIFICATE", Bytes: certDER}); err != nil {
		return errors.Wrap(err, "encode cert")
	}

	keyDER, err := x509.MarshalECPrivateKey(key)
	if err != nil {
		return errors.Wrap(err, "marshal key")
	}
	kf, err := os.OpenFile(keyFile, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return errors.Wrap(err, "open key file")
	}
	defer kf.Close()
	if err := pem.Encode(kf, &pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER}); err != nil {
		return errors.Wrap(err, "encode key")
	}

	return nil
}

func setGoproxyCA(cert tls.Certificate) {
	goproxy.GoproxyCa = cert
}

func caHandler(certFile string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-x509-ca-cert")
		w.Header().Set("Content-Disposition", `attachment; filename="salus-ca.crt"`)
		http.ServeFile(w, r, certFile)
	}
}
