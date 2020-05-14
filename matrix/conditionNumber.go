package matrix

// Norm of the passed matrix
func Norm(m *Matrix) float64 {
	norm := 0.0

	for i := 0; i < m.rows; i++ {
		cur := 0.0

		for j := 0; j < m.cols; j++ {
			if m.matrix[i][j] < 0.0 {
				cur += - m.matrix[i][j]
			} else {
				cur += m.matrix[i][j]
			}
		}

		if cur > norm {
			norm = cur
		}
	}

	return norm
}

// ConditionNumberOf passed matrix
func ConditionNumberOf(m *Matrix) float64 {
	inverse := Inverse(m)
	
	return Norm(&inverse) / Norm(m)
}