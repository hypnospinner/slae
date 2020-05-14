package matrix

// Matrix : some simple abstraction over a mathematical matrix
type Matrix struct {
	rows   int
	cols   int
	matrix [][]float64
}

// NewEmptySquare creates new square matrix filled with 0 of specified size
func NewEmptySquare(size int) Matrix {
	matrix := make([][]float64, size)

	for i := 0; i < size; i++ {
		matrix[i] = make([]float64, size)

		for j := 0; j < size; j++ {
			matrix[i][j] = 0
		}
	}

	return Matrix{
		rows:   size,
		cols:   size,
		matrix: matrix,
	}
}

// NewSingularSquare creates new singular square matrix of specified size
func NewSingularSquare(size int) Matrix {
	matrix := make([][]float64, size)

	for i := 0; i < size; i++ {
		matrix[i] = make([]float64, size)

		for j := 0; j < size; j++ {

			if i == j {
				matrix[i][j] = 1
			} else {
				matrix[i][j] = 0
			}
		}
	}

	return Matrix{
		rows:   size,
		cols:   size,
		matrix: matrix,
	}
}

// NewSingular creates new singular matrix with specified number of rows & cols
func NewSingular(rows int, cols int) Matrix {
	matrix := make([][]float64, rows)

	for i := 0; i < rows; i++ {
		matrix[i] = make([]float64, cols)

		for j := 0; j < cols; j++ {

			if i == j {
				matrix[i][j] = 1
			} else {
				matrix[i][j] = 0
			}
		}
	}

	return Matrix{
		rows:   rows,
		cols:   cols,
		matrix: matrix,
	}
}

// NewEmpty creates new matrix filled with 0 with specified number of rows & cols
func NewEmpty(rows int, cols int) Matrix {
	matrix := make([][]float64, rows)

	for i := 0; i < rows; i++ {
		matrix[i] = make([]float64, cols)

		for j := 0; j < cols; j++ {
			matrix[i][j] = 0
		}
	}

	return Matrix{
		rows:   rows,
		cols:   cols,
		matrix: matrix,
	}
}

// NewFrom creates new matrix from 2d float64 array
func NewFrom(matrix [][]float64) Matrix {
	return Matrix{
		rows:   len(matrix),
		cols:   len(matrix[0]),
		matrix: matrix,
	}
}

// NewRowFrom creates new matrix with one row by passed array or slice
func NewRowFrom(row []float64) Matrix {
	value := make([][]float64, 1)

	value[0] = row
	
	return Matrix {
		rows: 1,
		cols: len(row),
		matrix: value,
	}
}

// NewColFrom creates new matrix with one col by passed array or slice
func NewColFrom(col []float64) Matrix {
	value := make([][]float64, len(col))

	for i := 0; i < len(col); i++ {
		value[i] = make([]float64, 1)
		value[i][0] = col[i]
	}
	
	return Matrix {
		cols: 1,
		rows: len(col),
		matrix: value,
	}
}