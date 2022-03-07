package memorylearning

import "github.com/ruesier/complexMatrix"
import "math"
import "math/cmplx"

type Model struct {
	memory             []float64
	default_memory     []float64
	inputLayer         [][]float64
	outputLayer        [][]float64
	memoryLayer        [][]float64
	internalLayers     []complexMatrix.M
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
	for i := 0; i < len(data) - 1; i++ {
		data[i] = math.Tanh(data[i])
	}

	combined := complexMatrix.CombineIntoMutable([][]float64{data}, [][]float64{m.memory})

	for _, layer := range m.internalLayers {
		combined = combined.Dot(layer)
		combined = combined.Map(tanh_activation_complex)
	}

	dataMatrix, memMatrix := complexMatrix.Parts(combined)
	data = vectorDot(dataMatrix[0], m.outputLayer)
	data = data[:len(data)-1]
	for i := range data {
		data[i] = math.Tanh(data[i])
	}

	m.memory = vectorDot(memMatrix[0], m.memoryLayer)
	for i := 0; i < len(m.memory) - 1; i++ {
		m.memory[i] = math.Tanh(m.memory[i])
	}

	return data
}

func (m *Model) Reset() {
	m.memory = make([]float64, 0, len(m.default_memory))
	copy(m.memory, m.default_memory)
}

func tanh_activation_complex(val complex128, col int, row int, matrix complexMatrix.M) complex128 {
	magnitude, theta := cmplx.Polar(val)
	if W, H := matrix.Dim(); row == H - 1 || col == W - 1 {
		return cmplx.Rect(magnitude, theta)
	} else {
		return cmplx.Rect(math.Tanh(magnitude), theta)
	}
}