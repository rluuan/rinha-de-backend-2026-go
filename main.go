package main

import (
	"encoding/json"
	"log"
	"math"
	"net/http"
	"os"
	"time"
)

const fraudThreshold = 0.6

var (
	weights [14]float64
	bias    float64
	mccRisk map[string]float64
)

type request struct {
	Transaction struct {
		Amount       float64 `json:"amount"`
		Installments int     `json:"installments"`
		RequestedAt  string  `json:"requested_at"`
	} `json:"transaction"`
	Customer struct {
		AvgAmount      float64  `json:"avg_amount"`
		TxCount24h     int      `json:"tx_count_24h"`
		KnownMerchants []string `json:"known_merchants"`
	} `json:"customer"`
	Merchant struct {
		ID        string  `json:"id"`
		MCC       string  `json:"mcc"`
		AvgAmount float64 `json:"avg_amount"`
	} `json:"merchant"`
	Terminal struct {
		IsOnline    bool    `json:"is_online"`
		CardPresent bool    `json:"card_present"`
		KmFromHome  float64 `json:"km_from_home"`
	} `json:"terminal"`
	LastTransaction *struct {
		Timestamp     string  `json:"timestamp"`
		KmFromCurrent float64 `json:"km_from_current"`
	} `json:"last_transaction"`
}

func clamp(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func vectorize(r *request) [14]float64 {
	var v [14]float64

	v[0] = clamp(r.Transaction.Amount / 10000)
	v[1] = clamp(float64(r.Transaction.Installments) / 12)
	if r.Customer.AvgAmount > 0 {
		v[2] = clamp((r.Transaction.Amount / r.Customer.AvgAmount) / 10)
	}

	t, _ := time.Parse(time.RFC3339, r.Transaction.RequestedAt)
	v[3] = float64(t.UTC().Hour()) / 23
	v[4] = float64((int(t.UTC().Weekday())+6)%7) / 6

	if r.LastTransaction != nil {
		lastT, _ := time.Parse(time.RFC3339, r.LastTransaction.Timestamp)
		v[5] = clamp(t.Sub(lastT).Minutes() / 1440)
		v[6] = clamp(r.LastTransaction.KmFromCurrent / 1000)
	} else {
		v[5] = -1
		v[6] = -1
	}

	v[7] = clamp(r.Terminal.KmFromHome / 1000)
	v[8] = clamp(float64(r.Customer.TxCount24h) / 20)

	if r.Terminal.IsOnline {
		v[9] = 1
	}
	if r.Terminal.CardPresent {
		v[10] = 1
	}

	known := make(map[string]struct{}, len(r.Customer.KnownMerchants))
	for _, m := range r.Customer.KnownMerchants {
		known[m] = struct{}{}
	}
	if _, ok := known[r.Merchant.ID]; !ok {
		v[11] = 1
	}

	if risk, ok := mccRisk[r.Merchant.MCC]; ok {
		v[12] = risk
	} else {
		v[12] = 0.5
	}

	v[13] = clamp(r.Merchant.AvgAmount / 10000)
	return v
}

func fraudScore(v [14]float64) float64 {
	dot := bias
	for i, xi := range v {
		dot += weights[i] * xi
	}
	return sigmoid(dot)
}

func main() {
	data, err := os.ReadFile("data/mcc_risk.json")
	if err != nil {
		log.Fatalf("mcc_risk.json: %v", err)
	}
	if err := json.Unmarshal(data, &mccRisk); err != nil {
		log.Fatalf("parse mcc_risk.json: %v", err)
	}

	type weightsFile struct {
		W    [14]float64 `json:"w"`
		Bias float64     `json:"bias"`
	}
	wdata, err := os.ReadFile("weights.json")
	if err != nil {
		log.Fatalf("weights.json: %v", err)
	}
	var wf weightsFile
	if err := json.Unmarshal(wdata, &wf); err != nil {
		log.Fatalf("parse weights.json: %v", err)
	}
	weights = wf.W
	bias = wf.Bias

	mux := http.NewServeMux()

	mux.HandleFunc("GET /ready", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status":"ok"}`))
	})

	mux.HandleFunc("POST /fraud-score", func(w http.ResponseWriter, r *http.Request) {
		var req request
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		score := fraudScore(vectorize(&req))
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"approved":    score < fraudThreshold,
			"fraud_score": score,
		})
	})

	port := os.Getenv("PORT")
	if port == "" {
		port = "9999"
	}
	srv := &http.Server{
		Addr:        ":" + port,
		Handler:     mux,
		IdleTimeout: 65 * time.Second,
	}
	log.Printf("listening on :%s", port)
	log.Fatal(srv.ListenAndServe())
}
