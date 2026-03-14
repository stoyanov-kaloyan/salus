package main

import (
	"bytes"
	"image"
	"image/draw"
	"image/gif"
	"image/jpeg"
	"image/png"
	"io"
	"log/slog"
	"net/http"

	_ "golang.org/x/image/webp"
)

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

	img, format, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		slog.Warn("image: decode failed", "err", err, "content-type", ct)
		resp.Body = io.NopCloser(bytes.NewReader(data))
		return resp
	}

	if !shouldBlur(img, cfg) {
		resp.Body = io.NopCloser(bytes.NewReader(data))
		return resp
	}

	blurred := blurImage(img, cfg.BlurRadius)

	var buf bytes.Buffer
	outFormat := format
	if outFormat == "webp" {
		// golang.org/x/image has no WebP encoder; downgrade to JPEG.
		outFormat = "jpeg"
		resp.Header.Set("Content-Type", "image/jpeg")
	}

	switch outFormat {
	case "jpeg":
		err = jpeg.Encode(&buf, blurred, &jpeg.Options{Quality: 85})
	case "png":
		err = png.Encode(&buf, blurred)
	case "gif":
		err = gif.Encode(&buf, toNRGBA(blurred), nil)
	default:
		err = jpeg.Encode(&buf, blurred, &jpeg.Options{Quality: 85})
		resp.Header.Set("Content-Type", "image/jpeg")
	}

	if err != nil {
		slog.Warn("image: encode failed", "err", err, "format", outFormat)
		resp.Body = io.NopCloser(bytes.NewReader(data))
		return resp
	}

	resp.Body = io.NopCloser(bytes.NewReader(buf.Bytes()))
	resp.ContentLength = int64(buf.Len())
	resp.Header.Del("Content-Length") // net/http writes it from ContentLength

	slog.Debug("image: blurred", "format", outFormat, "bytes", buf.Len())
	return resp
}

// TODO: call detection API instead of always blurring.
func shouldBlur(_ image.Image, cfg *Config) bool {
	if cfg.DetectionAPIURL != "" {
		// TODO: POST image to detection API, parse response.
		return true
	}
	return true
}

// 3 passes of box blur approximates a Gaussian.
func blurImage(src image.Image, radius int) image.Image {
	rgba := toRGBA(src)
	for range 3 {
		boxBlurH(rgba, radius)
		boxBlurV(rgba, radius)
	}
	return rgba
}

func toRGBA(src image.Image) *image.RGBA {
	b := src.Bounds()
	dst := image.NewRGBA(b)
	draw.Draw(dst, b, src, b.Min, draw.Src)
	return dst
}

func toNRGBA(src image.Image) *image.NRGBA {
	b := src.Bounds()
	dst := image.NewNRGBA(b)
	draw.Draw(dst, b, src, b.Min, draw.Src)
	return dst
}

func boxBlurH(img *image.RGBA, radius int) {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	tmp := make([]byte, w*4)

	for y := 0; y < h; y++ {
		row := img.Pix[y*img.Stride : y*img.Stride+w*4]
		var rS, gS, bS, aS int
		count := radius + 1

		// Seed the window with left-edge replication.
		for i := 0; i <= radius; i++ {
			xi := clamp(i, 0, w-1) * 4
			rS += int(row[xi])
			gS += int(row[xi+1])
			bS += int(row[xi+2])
			aS += int(row[xi+3])
		}

		for x := 0; x < w; x++ {
			tmp[x*4] = byte(rS / count)
			tmp[x*4+1] = byte(gS / count)
			tmp[x*4+2] = byte(bS / count)
			tmp[x*4+3] = byte(aS / count)

			// Slide window: remove left, add right.
			xl := clamp(x-radius, 0, w-1) * 4
			xr := clamp(x+radius+1, 0, w-1) * 4
			rS += int(row[xr]) - int(row[xl])
			gS += int(row[xr+1]) - int(row[xl+1])
			bS += int(row[xr+2]) - int(row[xl+2])
			aS += int(row[xr+3]) - int(row[xl+3])
		}
		copy(row, tmp)
	}
}

func boxBlurV(img *image.RGBA, radius int) {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	tmp := make([]byte, h*4)

	for x := 0; x < w; x++ {
		var rS, gS, bS, aS int
		count := radius + 1

		for i := 0; i <= radius; i++ {
			yi := clamp(i, 0, h-1)*img.Stride + x*4
			rS += int(img.Pix[yi])
			gS += int(img.Pix[yi+1])
			bS += int(img.Pix[yi+2])
			aS += int(img.Pix[yi+3])
		}

		for y := 0; y < h; y++ {
			tmp[y*4] = byte(rS / count)
			tmp[y*4+1] = byte(gS / count)
			tmp[y*4+2] = byte(bS / count)
			tmp[y*4+3] = byte(aS / count)

			yt := clamp(y-radius, 0, h-1)*img.Stride + x*4
			yb := clamp(y+radius+1, 0, h-1)*img.Stride + x*4
			rS += int(img.Pix[yb]) - int(img.Pix[yt])
			gS += int(img.Pix[yb+1]) - int(img.Pix[yt+1])
			bS += int(img.Pix[yb+2]) - int(img.Pix[yt+2])
			aS += int(img.Pix[yb+3]) - int(img.Pix[yt+3])
		}

		for y := 0; y < h; y++ {
			yi := y*img.Stride + x*4
			img.Pix[yi] = tmp[y*4]
			img.Pix[yi+1] = tmp[y*4+1]
			img.Pix[yi+2] = tmp[y*4+2]
			img.Pix[yi+3] = tmp[y*4+3]
		}
	}
}

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
