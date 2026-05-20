package main

import (
	"bufio"
	"compress/gzip"
	"encoding/json"
	"log"
	"math"
	"os"
)

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func main() {
	f, err := os.Open("data/references.json.gz")
	if err != nil {
		log.Fatalf("open references.json.gz: %v", err)
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		log.Fatalf("gzip reader: %v", err)
	}
	defer gz.Close()

	var w [14]float64
	b := 0.0
	const lr = 0.01
	n := 0

	type record struct {
		Vector [14]float64 `json:"vector"`
		Label  string      `json:"label"`
	}

	dec := json.NewDecoder(bufio.NewReaderSize(gz, 1<<20))
	dec.Token()

	for dec.More() {
		var rec record
		if decErr := dec.Decode(&rec); decErr != nil {
			log.Printf("decode error at record %d: %v", n, decErr)
			continue
		}

		y := 0.0
		if rec.Label == "fraud" {
			y = 1.0
		}

		dot := b
		for i, xi := range rec.Vector {
			dot += w[i] * xi
		}

		grad := sigmoid(dot) - y
		for i, xi := range rec.Vector {
			w[i] -= lr * grad * xi
		}
		b -= lr * grad
		n++

		if n%500000 == 0 {
			log.Printf("treinando... %d registros", n)
		}
	}

	log.Printf("treinamento concluido: %d registros", n)

	type weightsFile struct {
		W    [14]float64 `json:"w"`
		Bias float64     `json:"bias"`
	}
	out, err := os.Create("weights.json")
	if err != nil {
		log.Fatalf("create weights.json: %v", err)
	}
	defer out.Close()

	enc := json.NewEncoder(out)
	enc.SetIndent("", "  ")
	if err := enc.Encode(weightsFile{W: w, Bias: b}); err != nil {
		log.Fatalf("encode weights: %v", err)
	}
	log.Println("pesos salvos em weights.json")
}
