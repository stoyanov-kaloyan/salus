package main

import (
	"bytes"
	"crypto/rand"
	"crypto/tls"
	"encoding/base64"
	"fmt"
	"net/http"
	"strings"
)

func buildSetupResponse(req *http.Request, cfg *Config, cert tls.Certificate) *http.Response {
	path := req.URL.Path
	switch path {
	case "/ios":
		return mobileConfigResponse(cert)
	case "/android":
		return derCertResponse(cert)
	case "/pem":
		return pemCertResponse(cert)
	default:
		return setupPageResponse(req)
	}
}

func mobileConfigResponse(cert tls.Certificate) *http.Response {
	der := cert.Certificate[0]
	b64 := base64.StdEncoding.EncodeToString(der)

	var indented strings.Builder
	for i := 0; i < len(b64); i += 64 {
		end := i + 64
		if end > len(b64) {
			end = len(b64)
		}
		indented.WriteString("        ")
		indented.WriteString(b64[i:end])
		indented.WriteString("\n")
	}

	plist := fmt.Sprintf(`<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>PayloadVersion</key><integer>1</integer>
  <key>PayloadUUID</key><string>%s</string>
  <key>PayloadType</key><string>Configuration</string>
  <key>PayloadIdentifier</key><string>io.salus.proxy.ca</string>
  <key>PayloadDisplayName</key><string>Salus Proxy CA</string>
  <key>PayloadDescription</key><string>Allows Salus proxy to inspect HTTPS traffic</string>
  <key>PayloadOrganization</key><string>Salus</string>
  <key>PayloadContent</key>
  <array>
    <dict>
      <key>PayloadType</key><string>com.apple.security.root</string>
      <key>PayloadVersion</key><integer>1</integer>
      <key>PayloadIdentifier</key><string>io.salus.proxy.ca.root</string>
      <key>PayloadUUID</key><string>%s</string>
      <key>PayloadDisplayName</key><string>Salus Proxy CA</string>
      <key>PayloadContent</key>
      <data>
%s      </data>
    </dict>
  </array>
</dict>
</plist>`, newUUID(), newUUID(), indented.String())

	body := []byte(plist)
	resp := &http.Response{
		Status:        "200 OK",
		StatusCode:    http.StatusOK,
		Proto:         "HTTP/1.1",
		ProtoMajor:    1,
		ProtoMinor:    1,
		Header:        make(http.Header),
		Body:          bodyFromBytes(body),
		ContentLength: int64(len(body)),
	}
	resp.Header.Set("Content-Type", "application/x-apple-aspen-config")
	resp.Header.Set("Content-Disposition", `attachment; filename="salus-proxy.mobileconfig"`)
	return resp
}

func derCertResponse(cert tls.Certificate) *http.Response {
	body := cert.Certificate[0]
	resp := &http.Response{
		Status:        "200 OK",
		StatusCode:    http.StatusOK,
		Proto:         "HTTP/1.1",
		ProtoMajor:    1,
		ProtoMinor:    1,
		Header:        make(http.Header),
		Body:          bodyFromBytes(body),
		ContentLength: int64(len(body)),
	}
	resp.Header.Set("Content-Type", "application/x-x509-ca-cert")
	resp.Header.Set("Content-Disposition", `attachment; filename="salus-proxy-ca.crt"`)
	return resp
}

func pemCertResponse(cert tls.Certificate) *http.Response {
	var sb strings.Builder
	sb.WriteString("-----BEGIN CERTIFICATE-----\n")
	b64 := base64.StdEncoding.EncodeToString(cert.Certificate[0])
	for i := 0; i < len(b64); i += 64 {
		end := i + 64
		if end > len(b64) {
			end = len(b64)
		}
		sb.WriteString(b64[i:end])
		sb.WriteString("\n")
	}
	sb.WriteString("-----END CERTIFICATE-----\n")
	body := []byte(sb.String())
	resp := &http.Response{
		Status:        "200 OK",
		StatusCode:    http.StatusOK,
		Proto:         "HTTP/1.1",
		ProtoMajor:    1,
		ProtoMinor:    1,
		Header:        make(http.Header),
		Body:          bodyFromBytes(body),
		ContentLength: int64(len(body)),
	}
	resp.Header.Set("Content-Type", "application/x-pem-file")
	resp.Header.Set("Content-Disposition", `attachment; filename="salus-proxy-ca.pem"`)
	return resp
}

func setupPageResponse(req *http.Request) *http.Response {
	ua := req.Header.Get("User-Agent")
	p := detectPlatform(ua)
	body := []byte(buildSetupHTML(p))
	resp := &http.Response{
		Status:        "200 OK",
		StatusCode:    http.StatusOK,
		Proto:         "HTTP/1.1",
		ProtoMajor:    1,
		ProtoMinor:    1,
		Header:        make(http.Header),
		Body:          bodyFromBytes(body),
		ContentLength: int64(len(body)),
	}
	resp.Header.Set("Content-Type", "text/html; charset=utf-8")
	return resp
}

type platform int

const (
	platformOther platform = iota
	platformIOS
	platformAndroid
)

func detectPlatform(ua string) platform {
	switch {
	case strings.Contains(ua, "iPhone") || strings.Contains(ua, "iPad"):
		return platformIOS
	case strings.Contains(ua, "Android"):
		return platformAndroid
	default:
		return platformOther
	}
}

func buildSetupHTML(p platform) string {
	type btn struct {
		href, label, style string
	}
	buttons := []btn{
		{"/ios", "iOS / iPadOS (.mobileconfig)", "primary"},
		{"/android", "Android (.crt)", "secondary"},
		{"/pem", "macOS / Windows / Linux (.pem)", "secondary"},
	}
	// Move the detected platform's button to the front.
	switch p {
	case platformIOS:
		// already first
	case platformAndroid:
		buttons[0], buttons[1] = buttons[1], buttons[0]
	default:
		buttons[0], buttons[2] = buttons[2], buttons[0]
	}

	var btnHTML strings.Builder
	for _, b := range buttons {
		cls := "btn"
		if b.style == "primary" {
			cls += " btn-primary"
		}
		fmt.Fprintf(&btnHTML, `<a href="%s" class="%s">%s</a>`+"\n", b.href, cls, b.label)
	}

	var instructions string
	switch p {
	case platformIOS:
		instructions = `<ol>
  <li>Tap <strong>iOS (.mobileconfig)</strong> above and confirm the download.</li>
  <li>Open <strong>Settings → General → VPN &amp; Device Management</strong>.</li>
  <li>Tap <em>Salus Proxy CA</em> → <strong>Install</strong>.</li>
  <li>Go to <strong>Settings → General → About → Certificate Trust Settings</strong>.</li>
  <li>Enable full trust for <em>Salus Proxy CA</em>.</li>
</ol>`
	case platformAndroid:
		instructions = `<ol>
  <li>Tap <strong>Android (.crt)</strong> above to download the certificate.</li>
  <li>Open <strong>Settings → Security → Install a certificate → CA certificate</strong>.</li>
  <li>Select the downloaded file and confirm installation.</li>
</ol>`
	default:
		instructions = `<ol>
  <li>Download the <strong>.pem</strong> certificate above.</li>
  <li><strong>macOS</strong>: Double-click the file → Keychain Access → set to <em>Always Trust</em>.</li>
  <li><strong>Windows</strong>: Double-click → Install Certificate → Trusted Root Certification Authorities.</li>
  <li><strong>Linux / Chrome</strong>: <code>sudo cp ca.pem /usr/local/share/ca-certificates/salus-proxy.crt &amp;&amp; sudo update-ca-certificates</code></li>
</ol>`
	}

	return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Salus Proxy Setup</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 600px; margin: 40px auto; padding: 0 20px; color: #222; }
  h1   { font-size: 1.6rem; margin-bottom: 0.3em; }
  p    { color: #555; }
  .btn { display: block; margin: 10px 0; padding: 14px 20px; border-radius: 8px;
         text-decoration: none; font-size: 1rem; text-align: center;
         background: #e8e8e8; color: #333; }
  .btn-primary { background: #0071e3; color: #fff; font-weight: 600; }
  .instructions { margin-top: 30px; background: #f5f5f5; border-radius: 8px;
                  padding: 16px 20px; }
  .instructions h2 { margin-top: 0; font-size: 1rem; }
  code { background: #e0e0e0; padding: 2px 5px; border-radius: 4px; font-size: 0.85em; }
</style>
</head>
<body>
<h1>Salus Proxy Setup</h1>
<p>Install the CA certificate on your device to allow the proxy to inspect HTTPS traffic.</p>
` + btnHTML.String() + `
<div class="instructions">
  <h2>Installation steps</h2>
` + instructions + `
</div>
</body>
</html>`
}

func bodyFromBytes(b []byte) *readCloser {
	return &readCloser{bytes.NewReader(b)}
}

type readCloser struct{ *bytes.Reader }

func (r *readCloser) Close() error { return nil }

func newUUID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	b[6] = (b[6] & 0x0f) | 0x40 // version 4
	b[8] = (b[8] & 0x3f) | 0x80 // variant bits
	return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x",
		b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}
