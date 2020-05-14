package matrix

import (
	"errors"
	"fmt"
)

// Copy returns new instance of the same matrix
func (m *Matrix) Copy() Matrix {
	matrix := make([][]float64, m.rows)

	for i := 0; i < m.rows; i++ {
		matrix[i] = make([]float64, m.cols)

		for j := 0; j < m.cols; j++ {
			matrix[i][j] = m.matrix[i][j]
		}
	}

	return Matrix{
		rows:   m.rows,
		cols:   m.cols,
		matrix: matrix,
	}
}

// TODO CopySubMatrix returns new instance of some part of the matrix
// func (m *Matrix) CopySubMatrix(fromRow int, fromCol int, toRow int, toCol int) Matrix {

// }

// PickRow returns a copy of specified row of matrix
func (m *Matrix) PickRow(row int) Matrix {
	rowMatrix := make([][]float64, 1)
	rowMatrix[0] = make([]float64, m.cols)

	for i := 0; i < m.cols; i++ {
		rowMatrix[0][i] = m.matrix[row][i]
	}

	return Matrix{
		rows:   1,
		cols:   m.cols,
		matrix: rowMatrix,
	}
}

// PickCol returns a copy of specified col of matrix
func (m *Matrix) PickCol(col int) Matrix {
	colMatrix := make([][]float64, m.rows)

	for i := 0; i < m.rows; i++ {
		colMatrix[i] = make([]float64, 1)
		colMatrix[i][0] = m.matrix[i][col]
	}

	return Matrix{
		rows:   1,
		cols:   m.cols,
		matrix: colMatrix,
	}
}

// Dot stands for dot product of 2 matrices
func Dot(left *Matrix, right *Matrix) (Matrix, error) {
	var product Matrix

	if left.cols != right.rows {
		return product, errors.New("Can't calculate dot product of 2 matrices with not matching size")
	}

	product.rows = left.rows
	product.cols = right.cols
	product.matrix = make([][]float64, product.rows)

	for i := 0; i < product.rows; i++ {
		product.matrix[i] = make([]float64, product.cols)

		for j := 0; j < product.cols; j++ {
			product.matrix[i][j] = 0

			for k := 0; k < left.cols; k++ {
				product.matrix[i][j] += left.matrix[i][k] * right.matrix[k][j]
			}
		}
	}

	return product, nil
}

// Print outputs matrix into standart fmt output
func (m *Matrix) Print() {

	for i := 0; i < m.rows; i++ {
		fmt.Print("[ ")

		for j := 0; j < m.cols; j++ {
			fmt.Printf(" %6.3f ", m.matrix[i][j])
		}

		fmt.Println(" ]")
	}
}

// SwapCols changes values of 2 columns in a matrix
func (m *Matrix) SwapCols(from int, to int) {
	for i := 0; i < m.rows; i++ {
		t := m.matrix[i][from]
		m.matrix[i][from] = m.matrix[i][to]
		m.matrix[i][to] = t
	}
}

// SwapRows changes values of 2 rows in a matrix
func (m *Matrix) SwapRows(from int, to int) {
	for i := 0; i < m.cols; i++ {
		t := m.matrix[from][i]
		m.matrix[from][i] = m.matrix[to][i]
		m.matrix[to][i] = t
	}
}

// Transpose for square matrix only
func (m *Matrix) Transpose() Matrix {
	result := m.Copy()

	for i := 0; i < result.rows; i++ {
		for j := i + 1; j < result.cols; j++ {
			temp := result.matrix[i][j]
			result.matrix[i][j] = result.matrix[j][i]
			result.matrix[j][i] = temp
		}
	}

	return result
}

