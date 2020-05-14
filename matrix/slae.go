package matrix

// Solve system of liner equations using LU decomposition with partial pivoting
// returns solution for initial matrix
func Solve(A *Matrix, b []float64) []float64 {
	L, U, P, _, _ := A.Factorize()

	x := make([]float64, len(b))
	y := make([]float64, len(b))

	bMatrixValue := make([][]float64, len(b))

	for i := 0; i < len(b); i++ {
		bMatrixValue[i] = make([]float64, 1)
		bMatrixValue[i][0] = b[i]
	}

	// reorder b
	bMatrix := Matrix{
		rows:   len(b),
		cols:   1,
		matrix: bMatrixValue,
	}

	bMatrixReordered, _ := Dot(P, &bMatrix)

	// solve Ly = b
	for k := 0; k < len(y); k++ {
		y[k] = bMatrixReordered.matrix[k][0]

		for j := 0; j < k; j++ {
			y[k] -= y[j] * L.matrix[k][j]
		}
	}

	// solve Ux = y
	for k := len(x) - 1; k >= 0; k-- {
		x[k] = y[k]

		for j := k + 1; j < len(x); j++ {
			x[k] -= x[j] * U.matrix[k][j]
		}

		x[k] /= U.matrix[k][k]
	}

	return x
}