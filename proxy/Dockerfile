FROM golang:1.24-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o proxy .

FROM gcr.io/distroless/static
COPY --from=builder /app/proxy /proxy
EXPOSE 8080
ENTRYPOINT ["/proxy"]
