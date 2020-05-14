package matrix

// Inverse matrix of passed
func Inverse(m *Matrix) Matrix {
	// considering matrix is n x n
	// we solve n = 1..k..n systems Ax = (0,...,1_k,...,0)

	result := NewEmptySquare(m.cols)

	_b := make([]float64, m.rows)

	for i := 0; i < len(_b); i++ {
		_b[i] = 0.0
	}

	for i := 0; i < m.rows; i++ {
		if i > 0 {
			_b[i-1] = 0.0
		}

		_b[i] = 1.0

		x := NewColFrom(Solve(m, _b))

		for j := 0; j < m.cols; j++ {
			result.matrix[i][j] = x.matrix[j][0]
		}
	}

	return result
}
