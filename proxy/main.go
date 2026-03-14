package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	cfg := loadConfig()

	logLevel := slog.LevelInfo
	if cfg.Verbose {
		logLevel = slog.LevelDebug
	}
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: logLevel})))

	cert, err := loadOrCreateCA(cfg.CACertFile, cfg.CAKeyFile)
	if err != nil {
		slog.Error("CA init failed", "err", err)
		os.Exit(1)
	}
	setGoproxyCA(cert)
	slog.Info("CA ready", "cert", cfg.CACertFile)

	proxy := newProxy(cfg, cert)

	mux := http.NewServeMux()
	mux.Handle("/ca.crt", caHandler(cfg.CACertFile))
	mux.Handle("/", proxy)

	srv := &http.Server{
		Addr:        cfg.Addr,
		Handler:     mux,
		ReadTimeout: 30 * time.Second,
		// WriteTimeout intentionally 0: CONNECT tunnels are long-lived.
		IdleTimeout: 120 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	go func() {
		slog.Info("proxy listening", "addr", cfg.Addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			slog.Error("server error", "err", err)
			os.Exit(1)
		}
	}()

	<-ctx.Done()
	slog.Info("shutting down")

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		slog.Error("shutdown error", "err", err)
	}
}
