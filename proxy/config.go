package main

import (
	"log/slog"
	"os"
	"strconv"
)

type Config struct {
	Addr            string
	CACertFile      string
	CAKeyFile       string
	DetectionAPIURL string
	SetupHost       string
	Verbose         bool
}

func loadConfig() *Config {
	return &Config{
		Addr:            envStr("PROXY_ADDR", ":8080"),
		CACertFile:      envStr("PROXY_CA_CERT", "certs/ca.crt"),
		CAKeyFile:       envStr("PROXY_CA_KEY", "certs/ca.key"),
		DetectionAPIURL: envStr("PROXY_DETECTION_API_URL", ""),
		SetupHost:       envStr("PROXY_SETUP_HOST", "salus.proxy"),
		Verbose:         envBool("PROXY_VERBOSE", false),
	}
}

func envStr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envBool(key string, def bool) bool {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	b, err := strconv.ParseBool(v)
	if err != nil {
		slog.Warn("invalid env var, using default", "key", key, "value", v, "default", def)
		return def
	}
	return b
}
