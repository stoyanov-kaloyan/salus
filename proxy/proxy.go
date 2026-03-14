package main

import (
	"crypto/tls"
	"log/slog"
	"mime"
	"net/http"
	"strings"

	"github.com/elazarl/goproxy"
)

func newProxy(cfg *Config, cert tls.Certificate) *goproxy.ProxyHttpServer {
	proxy := goproxy.NewProxyHttpServer()
	proxy.Verbose = cfg.Verbose

	// Handle setup page requests before MITM kicks in.
	setupHost := cfg.SetupHost
	proxy.OnRequest(goproxy.ReqHostIs(setupHost, setupHost+":80")).DoFunc(
		func(req *http.Request, ctx *goproxy.ProxyCtx) (*http.Request, *http.Response) {
			return nil, buildSetupResponse(req, cfg, cert)
		},
	)

	proxy.OnRequest().HandleConnect(goproxy.AlwaysMitm)
	proxy.OnRequest().DoFunc(func(req *http.Request, ctx *goproxy.ProxyCtx) (*http.Request, *http.Response) {
		// 	slog.Info("request", "method", req.Method, "addr", req.RemoteAddr, "headers", req.Header)
		return req, nil
	})

	var imageResponse goproxy.RespConditionFunc = func(resp *http.Response, ctx *goproxy.ProxyCtx) bool {
		if resp == nil {
			return false
		}
		mediaType, _, _ := mime.ParseMediaType(resp.Header.Get("Content-Type"))
		slog.Info("media-type", mediaType)
		return strings.HasPrefix(mediaType, "image/")
	}
	proxy.OnResponse(imageResponse).DoFunc(func(resp *http.Response, ctx *goproxy.ProxyCtx) *http.Response {
		return processImage(resp, cfg)
	})

	return proxy
}
