package memorylearning

import "github.com/ruesier/complexMatrix"

type Model struct {
	memory         []float64
	default_memory []float64
	inputLayer     [][]float64
	outputLayer    [][]float64
	memoryLayer    [][]float64
	internalLayers []complexMatrix.M
	sigmoid        func(complex128, int, int, complexMatrix.M) complex128
}

func vectorDot(vec []float64, matrix [][]float64) []float64 {
	result := make([]float64, len(matrix[0]))
	for i := range result {
		for j := range matrix {
			result[i] += vec[j] * matrix[j][i]
		}
	}
	return result
}

func (m *Model) Predict(data []float64) []float64 {
	if len(data)+1 != len(m.inputLayer) {
		panic("input data is incorrect size for prediction")
	}
	data = append(data, 1)
	data = vectorDot(data, m.inputLayer)

	combined := complexMatrix.CombineIntoMutable([][]float64{data}, [][]float64{m.memory})

	for _, layer := range m.internalLayers {
		combined = combined.Dot(layer)
		combined = combined.Map(m.sigmoid)
	}

	dataMatrix, memMatrix := complexMatrix.Parts(combined)
	data = vectorDot(dataMatrix[0], m.outputLayer)
	data = data[:len(data)-1]

	m.memory = vectorDot(memMatrix[0], m.memoryLayer)

	return data
}

func (m *Model) Reset() {
	m.memory = make([]float64, 0, len(m.default_memory))
	copy(m.memory, m.default_memory)
}
