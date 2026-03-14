package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"io"
	"log/slog"
	"mime/multipart"
	"net/http"

	_ "embed"
	_ "golang.org/x/image/webp"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
)

//go:embed static/cover.jpg
var coverData []byte

func processImage(resp *http.Response, cfg *Config) *http.Response {
	if resp == nil || resp.Body == nil {
		return resp
	}

	ct := resp.Header.Get("Content-Type")
	// Strip Content-Encoding — goproxy already decoded it transparently.
	resp.Header.Del("Content-Encoding")

	data, err := io.ReadAll(io.LimitReader(resp.Body, 50<<20)) // 50 MB limit
	resp.Body.Close()
	if err != nil {
		slog.Warn("image: read body failed", "err", err)
		resp.Body = io.NopCloser(bytes.NewReader(data))
		return resp
	}

	_, format, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		slog.Warn("image: decode failed", "err", err, "content-type", ct)
		resp.Body = io.NopCloser(bytes.NewReader(data))
		return resp
	}

	if !shouldReplace(data, format, cfg) {
		resp.Body = io.NopCloser(bytes.NewReader(data))
		return resp
	}

	resp.Body = io.NopCloser(bytes.NewReader(coverData))
	resp.ContentLength = int64(len(coverData))
	resp.Header.Del("Content-Length")
	resp.Header.Set("Content-Type", "image/jpeg")

	slog.Debug("image: replaced with cover", "bytes", len(coverData))
	return resp
}

func shouldReplace(data []byte, format string, cfg *Config) bool {
	if cfg.DetectionAPIURL == "" {
		return false
	}

	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	fw, err := mw.CreateFormFile("file", fmt.Sprintf("image.%s", format))
	if err != nil {
		slog.Warn("detection: create form file failed", "err", err)
		return false
	}
	if _, err = fw.Write(data); err != nil {
		slog.Warn("detection: write form file failed", "err", err)
		return false
	}
	mw.Close()

	resp, err := http.Post(cfg.DetectionAPIURL, mw.FormDataContentType(), &body) //nolint:noctx
	if err != nil {
		slog.Warn("detection: request failed", "err", err)
		return false
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		slog.Warn("detection: non-200 response", "status", resp.StatusCode)
		return false
	}

	var result struct {
		IsTarget bool `json:"is_target"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		slog.Warn("detection: decode response failed", "err", err)
		return false
	}

	slog.Debug("detection: result", "is_target", result.IsTarget)
	return result.IsTarget
}
