package matrix

// Det stands for determinant of matrix
func (m *Matrix) Det() float64 {
	_, U, P, _, _ := m.Factorize()

	det := 1.0
	swaps:= 0
	for i := 0; i < U.rows; i++ {
		det *= U.matrix[i][i]

		if P.matrix[i][i] == 0 {
			swaps++
		}
	}

	if (swaps / 2) % 2 != 0 {
		det *= -1
	}

	return det
}