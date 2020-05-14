package main

import (
	"fmt"

	"github.com/hypnospinner/slae/matrix"
)

func main() {
	A := matrix.NewFrom([][]float64{
		{3.0, 1.0, 2.0},
		{5.0, 7.0, 5.0},
		{1.0, 2.0, 3.0}})

	L, U, P, Q, err := A.Factorize()

	if err != nil {
		return
	}

	fmt.Println("L: ")
	L.Print()

	fmt.Println("U: ")
	U.Print()

	fmt.Println("P: ")
	P.Print()

	fmt.Println("Q: ")
	Q.Print()

	fmt.Println("A: ")
	A.Print()

	fmt.Println("LU: ")
	LU, err := matrix.Dot(L, U)

	if err == nil {
		LU.Print()
	}

	fmt.Println("PA: ")
	PA, err := matrix.Dot(P, &A)

	if err == nil {
		PA.Print()
	}

	b := []float64{4.0, 4.0, 4.0}

	x := matrix.NewColFrom(matrix.Solve(&A, b))
	fmt.Println("x:")
	x.Print()

	calculatedB, _ := matrix.Dot(&A, &x)

	fmt.Println("Calculated b:")
	calculatedB.Print()

	E := matrix.NewSingularSquare(3)

	fmt.Printf("det A = %6.3f\n", A.Det())
	fmt.Printf("det E = %6.3f\n", E.Det())

	A_inverse := matrix.Inverse(&A)

	fmt.Println("Inverse A matrix:")
	A_inverse.Print()
}
