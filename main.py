import math
import operator
import numpy as np

# SWAP OPERATIONS

# swap rows
def swapr(matrix, fr, to):
    if fr != to:
        copy = np.copy(matrix)
        matrix[fr] = copy[to]
        matrix[to] = copy[fr]
    return matrix

# swap columns
def swapc(matrix, fr, to):
    if(fr != to):
        copy = np.copy(matrix)
        matrix[::, fr] = copy[::, to]
        matrix[::, to] = copy[::, fr]
    return matrix

def partiallyPivot(matrix):
    copy = abs(np.copy(matrix))

    P, Q = np.eye(len(matrix)), np.eye(len(matrix))

    for k in range(len(matrix)-1):
        # find max element among element in column under diagonal
        i, val = max(enumerate(copy[k::, k]), key=operator.itemgetter(1))

        copy = swapr(copy, k, i)
        P = swapr(P, k, i)

    return P, Q



# factorization for square matrix with partial pivoting
def factorize(matrix):
    # initialize:
    # U as copy of matrix
    # L as empty matrix
    # P, Q as E
    U = np.copy(matrix)
    L = np.zeros((len(matrix), len(matrix)))
    P, Q = partiallyPivot(matrix)
    
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            L[i,0] = matrix[i,0] / U[0,0]

            for k in range(i):
                U[i,j] -= L[i,k] * U[k,j]

            if i > j:
                L[j,i] = 0
            else:
                L[j,i] = matrix[j,i]

                for k in range(i):
                    L[j,i] -= L[j,k] * U[k,i]

                L[j,i] = L[j,i] / U[i,i]

    return L, U, P, Q

def solveSLAE(matrix, b):
    L, U, P, Q = factorize(matrix)
    x = np.zeros(len(b))    
    y = np.zeros(len(b))    

    pb = np.dot(P, b.T)

    for k in range(len(y)):
        y[k] = pb[k]

        for j in range(k):
            y[k] -= y[j] * L[k,j]

    for k in range(len(x)-1, -1, -1):
        x[k] = y[k]

        for j in range(k+1, len(x)):
            x[k] -= x[j] * U[k,j]

        x[k] /= U[k,k]
    
    return x

def det(matrix):
    # multiplication of elements on diagonal of U
    L, U, P, Q = factorize(matrix)
    
    return U.diagonal().prod()
    

def inverse(matrix):
    # AA^{-1} = E => we can solve Ax = (0,...,1_k,...0) n times to get inverse matrix
    E = np.eye(len(matrix))
    result = np.zeros((len(matrix),len(matrix)))

    for i in range(len(matrix)):
        result[i] = solveSLAE(matrix, E[i]).T
    
    return result.T

def norm(matrix):
    return max(abs(matrix[i]).sum() for i in range(len(matrix)))

def cond(matrix):
    return norm(matrix) * norm(inverse(matrix))
    
def solveIncompatibleSLAE(matrix, b):
    L, U, P, Q = factorize(matrix)
    
    rank = len(matrix)
    absU = abs(U)

    # python is not really accurate so I pick some very small number instead of pure 0
    while absU[rank-1].sum() < pow(10, -10):
        rank -= 1
    
    print(rank)

    row = 0

    y = np.zeros(len(b))    

    pb = np.dot(P, b.T)

    for k in range(len(y)):
        y[k] = pb[k]

        for j in range(k):
            y[k] -= y[j] * L[k,j]
    
    extended = np.column_stack((U, y))
    
    while abs(extended[row]).sum() > pow(10, -14):
        row += 1
        if row == len(matrix):
            break

    if rank == row:
        # case for compatible system
        x = np.zeros(len(matrix))
        x[rank - 1] = y[rank - 1] / U[rank - 1, rank - 1]
        
        for k in range(rank - 2, -1, -1):
            # z_k = (y_k - sum_{i = k+1}^{rank} U_k_i * z_i) / U_k_k
            x[k] = (y[k] - np.dot(U[k, k + 1:], x[k + 1:].T)) / U[k,k]

        # LU factorization is performed with partial pivoting so we can omit Qz there 
        return x
    else:
        # case for incompatible system
        return 'incompatible'

def factorizeQR(matrix):
    Q = np.zeros((len(matrix), len(matrix)))
    R = np.zeros((len(matrix), len(matrix)))
    a = np.copy(matrix)
    
    for k in range(len(matrix)):

        R[k, k] = math.sqrt(sum(pow(a[::, k], 2)))
        Q[::, k] = a[::, k] / R[k, k]

        for j in range(k+1, len(matrix)):
            R[k, j] = np.dot(Q[::, k].T, a[::, j])
            a[::, j] = a[::, j] - Q[::, k] * R[k, j]

    return Q, R


def solveSLAEQR(matrix, b):

    Q, R = factorizeQR(matrix)
    pb = np.dot(Q.T, b)

    x = np.zeros(len(R))

    x[len(R)-1] = pb[len(R)-1] / R[len(R)-1, len(R)-1]

    for k in range(len(R)-2, -1, -1):
        x[k] = (pb[k] - np.dot(R[k, k+1:],x[k+1:].T)) / R[k][k]

    return x

# not double penetration but diagonally prevalent
def makeDPMatrix(size, rank):
    # build some matrix
    # than put relatively huge values on diagonal intentionally
    matrix = np.array(np.random.randint(-rank, rank, size=(size, size)), float)

    np.fill_diagonal(matrix, 0)
    
    for i in range(size):
        matrix[i, i] = np.random.randint(1, 2 * rank) + sum(abs(matrix[i]))

    return matrix


def Jacobi(matrix, b):
    D = np.diag(np.diag(matrix))
    
    B = np.eye(len(matrix)) - np.dot(np.linalg.inv(D), matrix)
    
    d = np.dot(np.linalg.inv(D), b)
    
    Bnorm = norm(B)
    
    eps = pow(10, -9) * (1 - Bnorm) / Bnorm
    
    meas = d
    x = np.dot(B, meas) + d
    
    posterior = 1
    
    while max(abs(x-meas)) > eps:
        meas = x
        x = np.dot(B, x) + d
        posterior += 1

    prior = math.ceil((np.log(eps) + np.log(1 - Bnorm) - np.log(max(abs(d)))) / np.log(Bnorm))

    return x, prior, posterior

def Seidel(matrix, b):
    D = np.diag(np.diag(matrix))
    U = np.zeros((len(matrix), len(matrix)))
    L = np.zeros((len(matrix), len(matrix)))

    for i in range(len(matrix)-1):
        U[i, i+1::] = matrix[i, i+1::]
        L[i+1, :i+1] = matrix[i+1, :i+1]

    B = (-1)*np.dot(np.linalg.inv(L+D), U)
    meas = np.dot(np.linalg.inv(L+D), b)
    
    Bnorm = norm(B)
    
    eps = pow(10, -9) * (1 - Bnorm) / Bnorm
    
    xk = meas
    xkn = np.dot(B, xk) + meas
    
    posterior = 1
    
    while max(abs(xkn-xk)) > eps:
        xk = xkn
        xkn = np.dot(B, xkn) + meas
        posterior += 1

    prior = math.ceil((np.log(eps) + np.log(1 - Bnorm) - np.log(max(abs(meas)))) / np.log(Bnorm))

    return xkn, prior, posterior


######################################### output & testing #########################################

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

n = 5
A = np.float64(np.random.randint(-25,26,size=(n,n)))
#A = np.array([[1, 2, 1],
#              [2, 1, 1],
#              [1, -1, 2],], float)

L, U, P, Q = factorize(A)
LU = np.dot(L,U)
PAQ = np.dot(np.dot(P, A), Q)

print("A:")
print(A)

print("L:")
print(L)

print("U:")
print(U)

print("P:")
print(P)

print("Q:")
print(Q)

print("LU:")
print(LU)

print("PAQ:")
print(PAQ)

#b = np.array([4,4,4], float)
b = np.float64(np.random.randint(-25,26,size=(n)))

x = solveSLAE(A,b)

print("b:")
print(b)

print("x:")
print(x)

print("Ax:")
print(np.dot(A,x.T))

invA = inverse(A)

print("A^{-1}:")
print(invA)

AA_ = np.dot(A, invA)
_AA = np.dot(invA, A)

print("AA_{-1}:")
print(AA_)

print("A_{-1}A:")
print(_AA)

det = det(A)

print("detA:")
print(det)

cond = cond(A)

print("condA:")
print(cond)


#C = np.array([[1, 3, 5],
#              [1, -2, 3],
#              [2, 11, 12]], float)

#f = np.array([1, 2, 4], float)

C = np.array(A)
for j in range(1,n):
    C[:, j] = (2 - j) * C[:, 0] + (j - 1) * C[:, 2]

f = (2 - n) * C[:, 0] + (n - 1) * C[:, 2]

incx = solveIncompatibleSLAE(C, f)

print("C:")
print(C)
print("f: ")
print(f)
print("incompatible SLAE solution Cx = f: " + incx)

# B = np.array([  [9, 1, 0, 0],
#                 [1, 9, 1, 0],
#                 [0, 1, 9, 1],
#                 [0, 0, 1, 9]], float)
#
# e = np.array([1, 2, 4, 1], float)
#
# print("B:")
# print(B)
#
# print("e:")
# print(e)

Q, R = factorizeQR(A)

print("Q:")
print(Q)
print("R:")
print(R)

print("QR: ")
print(np.dot(Q,R))

qrx = solveSLAEQR(A, b)
print("QR solution: ")
print(qrx)

print("b: ", b)

print("Ax: ")
print(np.dot(A, qrx))

DP = makeDPMatrix(5, 20)

k = np.array(np.random.randint(-20, 20, 5), float)

print("DP:")
print(DP)

print("k:")
print(k)

print("Jacobi:")

jx, jprior, jposterior = Jacobi(DP, k)

print("prior: ")
print(jprior)

print("posterior")
print(jposterior)

print("x: ")
print(jx)
print("DBx: ")
print(np.dot(DP, jx))

print("Seidel:")

sx, sprior, sposterior = Seidel(DP, k)

print("prior: ")
print(sprior)

print("posterior")
print(sposterior)

print("x: ")
print(sx)
print("DBx: ")
print(np.dot(DP, sx))