import math
import operator
import numpy as np

# SWAP OPERATIONS

# swap rows


def swapr(matrix, fr, to):
    if fr != to:
        copy = np.copy(matrix)
        matrix[fr], matrix[to] = copy[to],  copy[fr]
    return matrix

# swap columns


def swapc(matrix, fr, to):
    if(fr != to):
        copy = np.copy(matrix)
        matrix[::, fr], matrix[::, to] = copy[::, to], copy[::, fr]
    return matrix

# swap col and row at the same time


def swapcr(matrix, frr, tor, frc, toc):
    matrix = swapr(matrix, frr, tor)
    matrix = swapc(matrix, frc, toc)
    return matrix


def pivot(L, U, P, Q, d):
    absU = abs(U)

    if absU[d:, d:].sum() < 1e-10:
        return False, L, U, P, Q

    i, j = np.where(absU[d:, d:] == absU[d:, d:].max())
    i[0] += d
    j[0] += d

    L = swapcr(L, i[0], d, j[0], d)
    U = swapcr(U, i[0], d, j[0], d)
    P = swapr(P, i[0], d)
    Q = swapc(Q, j[0], d)

    return True, L, U, P, Q


def factorize2(matrix):
    l = len(matrix)
    U = np.copy(matrix)
    L = np.zeros((l, l))
    P, Q = np.eye(l), np.eye(l)

    for i in range(l-1):

        success, L, U, P, Q = pivot(L, U, P, Q, i)

        if success == False:
            break

        T = np.eye(l)
        for k in range(i+1, l):
            L[k, i] = U[k, i] / U[i, i]
            T[k, i] = (-1) * L[k, i]

        print(f"\n{T}\n")
        U = np.dot(T, U)

    L = L + np.eye(l)

    return L, U, P, Q


def solveSLAE(matrix, b):
    L, U, P, Q = factorize2(matrix)
    x = np.zeros(len(b))
    y = np.zeros(len(b))

    pb = np.dot(P, b)

    l = len(pb)

    # Ly = Pb
    for k in range(l):
        y[k] = pb[k]

        for j in range(k):
            y[k] -= y[j] * L[k, j]

    # Ux = y
    for k in range(l-1, -1, -1):
        x[k] = y[k]

        for j in range(k+1, l):
            x[k] -= x[j] * U[k, j]

        x[k] /= U[k, k]

    return np.dot(Q, x)


def det(matrix):
    _, U, _, _ = factorize2(matrix)

    return U.diagonal().prod()


def inverse(matrix):
    E = np.eye(len(matrix))
    result = np.zeros((len(matrix), len(matrix)))

    for i in range(len(matrix)):
        result[i] = solveSLAE(matrix, E[i]).T

    return result.T


def norm(matrix):
    return max(abs(matrix[i]).sum() for i in range(len(matrix)))


def cond(matrix):
    return norm(matrix) * norm(inverse(matrix))


def incompatibleSolve(matrix, b):
    L, U, P, Q = factorize2(matrix)

    l = len(matrix)
    absU = abs(U)
    rank = l

    for i in range(l):
        if absU[i].sum() < 1e-10:
            rank -= 1

    pb = np.dot(P, b)
    y = np.zeros(l)

    for k in range(l):
        y[k] = pb[k]

        for j in range(k):
            y[k] -= y[j] * L[k, j]

    extended = np.column_stack((U, y))
    erank = l

    for i in range(l):
        if extended[i].sum() < 1e-10:
            erank -= 1

    if erank != rank:
        return False, None

    x = np.zeros(l)

    for k in range(l-1, -1, -1):
        x[k] = y[k]

        for j in range(k+1, l):
            x[k] -= x[j] * U[k, j]

        x[k] /= U[k, k]

    return True, np.dot(Q, x)


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
        x[k] = (pb[k] - np.dot(R[k, k+1:], x[k+1:].T)) / R[k][k]

    return x


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

    prior = math.ceil((np.log(eps) + np.log(1 - Bnorm) -
                       np.log(max(abs(d)))) / np.log(Bnorm))

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

    prior = math.ceil((np.log(eps) + np.log(1 - Bnorm) -
                       np.log(max(abs(meas)))) / np.log(Bnorm))

    return xkn, prior, posterior


######################################### output & testing #########################################

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

n = 5
A = np.float64(np.random.randint(-25, 26, size=(n, n)))
b = np.float64(np.random.randint(-25, 26, size=(n)))
# A = np.array([[1, 2, 1],
#              [2, 1, 1],
#              [1, -1, 2],], float)

L, U, P, Q = factorize2(A)
LU = np.dot(L, U)
PAQ = np.dot(np.dot(P, A), Q)

print("A:")
print(A)

print("\nL:")
print(L)

print("\nU:")
print(U)

print("\nP:")
print(P)

print("\nQ:")
print(Q)

print("\nLU:")
print(LU)

print("\nPAQ:")
print(PAQ)

print(f"\nLU - PAQ:\n{LU - PAQ}")

x = solveSLAE(A, b)
print(f"\nAx - b = {np.dot(A,x) - b}")

invA = inverse(A)
print(f"\nA^{-1}:\n{invA}")

AA_ = np.dot(A, invA)
_AA = np.dot(invA, A)
print(f"\nAA_{-1}:\n{AA_}")
print(f"\nA_{-1}A:\n{_AA}")

det = det(A)
print(f"\ndetA:\n {det}")

cond = cond(A)
print(f"\ncondA:\n{cond}")

C = np.array(A)
for j in range(1, n):
    C[:, j] = (2 - j) * C[:, 0] + (j - 1) * C[:, 2]

f = (2 - n) * C[:, 0] + (n - 1) * C[:, 2]

compatible, x = incompatibleSolve(C, f)

print(f"\nCompatible: {compatible}")

if compatible:
    print(f"\n Cx - f:\n{np.dot(C, x) - f}")

# # B = np.array([  [9, 1, 0, 0],
# #                 [1, 9, 1, 0],
# #                 [0, 1, 9, 1],
# #                 [0, 0, 1, 9]], float)
# #
# # e = np.array([1, 2, 4, 1], float)
# #
# # print("B:")
# # print(B)
# #
# # print("e:")
# # print(e)

# Q, R = factorizeQR(A)

# print("Q:")
# print(Q)
# print("R:")
# print(R)

# print("QR: ")
# print(np.dot(Q, R))

# qrx = solveSLAEQR(A, b)
# print("QR solution: ")
# print(qrx)

# print("b: ", b)

# print("Ax: ")
# print(np.dot(A, qrx))

# DP = makeDPMatrix(5, 20)

# k = np.array(np.random.randint(-20, 20, 5), float)

# print("DP:")
# print(DP)

# print("k:")
# print(k)

# print("Jacobi:")

# jx, jprior, jposterior = Jacobi(DP, k)

# print("prior: ")
# print(jprior)

# print("posterior")
# print(jposterior)

# print("x: ")
# print(jx)
# print("DBx: ")
# print(np.dot(DP, jx))

# print("Seidel:")

# sx, sprior, sposterior = Seidel(DP, k)

# print("prior: ")
# print(sprior)

# print("posterior")
# print(sposterior)

# print("x: ")
# print(sx)
# print("DBx: ")
# print(np.dot(DP, sx))
