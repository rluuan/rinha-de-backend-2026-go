FROM golang:1.22-alpine AS build
WORKDIR /app
COPY go.mod .
COPY main.go .
COPY train/ ./train/
COPY data/ ./data/
RUN go run ./train && go build -o server .

FROM alpine:3.19
WORKDIR /app
COPY --from=build /app/server .
COPY --from=build /app/weights.json .
COPY --from=build /app/data/mcc_risk.json ./data/
EXPOSE 9999
CMD ["./server"]
