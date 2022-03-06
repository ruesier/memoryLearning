package memorylearning

type Model interface {
	Predict(data []float64) []float64
	Reset()
}
