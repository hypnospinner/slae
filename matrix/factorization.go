package matrix

import (
	"errors"
)

// AbsFloat64 returns absolute value of float64 type
func AbsFloat64(input float64) float64 {
	if input < 0.0 {
		return -input
	}

	return input
}

// DiagonalPrevalent returns a matrix with rows swapped in the way we have
// greatest by absolute values in column be diagonal
func (m *Matrix) DiagonalPrevalent() (*Matrix, *Matrix, error) {

	var diagonalSize int
	P := NewSingular(m.rows, m.cols)
	a := m.Copy()

	if m.cols > m.rows {
		diagonalSize = m.rows
	} else {
		diagonalSize = m.cols
	}

	for i := 0; i < diagonalSize; i++ {
		// go through all the columns & pick the element on the main diagonal
		max, row := AbsFloat64(m.matrix[i][i]), i

		// go through each element of the column lower than diagonal one & find the greatest by absolute value
		for j := i + 1; j < m.rows; j++ {
			if max < AbsFloat64(m.matrix[j][i]) {
				max, row = AbsFloat64(m.matrix[j][i]), j
			}
		}

		if max == 0.0 {
			return nil, nil, errors.New("Can't build diagonally prevalent matrix")
		}

		if i != row {
			P.SwapRows(i, row)
			a.SwapRows(i, row)
		}
	}

	return &a, &P, nil
}

// Factorize returns result of LU factorization of matrix with partial pivoting
func (a *Matrix) Factorize() (*Matrix, *Matrix, *Matrix, *Matrix, error) {
	L := NewEmpty(a.rows, a.cols)
	U := NewEmpty(a.rows, a.cols)
	Q := NewSingular(a.rows, a.cols)

	m, P, err := a.DiagonalPrevalent()

	if err != nil {
		return nil, nil, nil, nil, err
	}

	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			U.matrix[0][i] = m.matrix[0][i]
			L.matrix[i][0] = m.matrix[i][0] / U.matrix[0][0]

			U.matrix[i][j] = m.matrix[i][j]

			for k := 0; k < i; k++ {
				U.matrix[i][j] -= L.matrix[i][k] * U.matrix[k][j]
			}

			if i > j {
				L.matrix[j][i] = 0
			} else {
				L.matrix[j][i] = m.matrix[j][i]

				for k := 0; k < i; k++ {
					L.matrix[j][i] -= L.matrix[j][k] * U.matrix[k][i]
				}

				L.matrix[j][i] /= U.matrix[i][i]
			}
		}
	}

	return &L, &U, P, &Q, nil
}