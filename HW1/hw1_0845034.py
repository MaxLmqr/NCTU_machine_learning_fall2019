#       HOMEWORK 1      #
#   Max LEMASQUERIER    #
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


# INPUT PARAMETERS
input_file = open("input.txt", "r")  # load input file
lbda = int(input("lbda : "))                       # regularization parameter
n = int(input("number of basis : "))                        # number of polynomial bases


########## FUNCTION DEFINITION ##########
# Transpose an array, we could have use np.transpose(M)...
def transpose(M):
    # Because np.shape return only one dimension when M is a line vector
    # I had to do a specific case.
    if len(np.shape(M)) == 1:
        tM = np.zeros((len(M), 1))
        for i in range(len(M)):
            tM[i] = M[i]
    # General case, works with a column vector because np.shape returns 2 dimensions
    else:
        m, n = np.shape(M)
        tM = np.zeros((n, m))
        for i in range(m):
            for j in range(n):
                tM[j][i] = M[i][j]
    return tM

# Multiplie two matrices A and B into a matrix C
# We could have use np.dot(A,B)...
def multiplie(A, B):
    if len(np.shape(A)) == 1:
        C = np.zeros((1, np.shape(B)[1]))
        for j in range(len(A)):
            for k in range(np.shape(B)[0]):
                C[j] += A[k]*B[k][j]
        return C
    elif np.shape(A)[1] == np.shape(B)[0]:
        C = np.zeros((np.shape(A)[0], np.shape(B)[1]))
        for i in range(np.shape(A)[0]):
            for j in range(np.shape(B)[1]):
                for k in range(np.shape(A)[1]):
                    C[i][j] += A[i][k]*B[k][j]
        return C
    else:
        raise Exception('Wrond matrices dimensions')

# Function to compute the LU decomposition
# Takes a matrix as argument
# Output Two matrices, L and U.
# This works only if A is non-singular
def LUdecomposition(U):
    n = np.shape(A)[1]
    L = np.eye((n))
    for i in range(n):
        if U[i][i] == 0:                # If diagonal coefficient is equal to 0
            for k in range(i+1, n):  # check the lines below
                if U[k][i] != 0:
                    U[[k, i]] = U[[i, k]]     # Swap lines
        # Diagonal coefficient is now != 0
        for j in range(i):
            L[i][j] = U[i][j]/U[j][j]
            U[i] = U[i] - (U[i][j]/U[j][j])*U[j]  # cancel coefficient U[i][j]
    return L, U
########## END OF FUNCTION DEFINITION ##########

# Reading the file
data = input_file.read().splitlines()
X = np.empty(len(data))
Y = np.empty(len(data))

# Setting up X and Y thanks to the set of data
for i in range(len(data)):
    current = data[i].split(',')
    X[i] = float(current[0])
    Y[i] = float(current[1])


# define curve coefficient array
curve_coefficient = np.zeros(n)

# Define matrix A
A = np.zeros((len(X), n))

# Fill matrix A
for i in range(n):
    A[:, i] = X**i

# Compute A transpose
At = transpose(A)

# Compute At * A + lambda*I and decompose it into L*U
U = multiplie(At, A) + lbda*np.eye(n)
L, U = LUdecomposition(U)


# Solve system to compute curve_coefficient
b = multiplie(At, transpose(Y))
# first system (solve L*X' = B') with X' = U*X and B = At*b
xtemp = np.zeros(len(L))
for i in range(len(xtemp)):
    xtemp[i] = b[i]
    for j in range(i):
        xtemp[i] = xtemp[i] - L[i][j]*xtemp[j]
    xtemp[i] = xtemp[i]/L[i][i]

# second system (solve U*X = X') to find X 
for i in range(len(curve_coefficient)):
    curve_coefficient[n-i-1] = xtemp[n-i-1]
    for j in range(i):
        curve_coefficient[n-i-1] = curve_coefficient[n-i-1] - \
            U[n-i-1][n-j-1]*curve_coefficient[n-j-1]
    curve_coefficient[n-i-1] = curve_coefficient[n-i-1]/U[n-i-1][n-i-1]

# Curve coefficient are now computed.

# Compute data to plot
Abs = np.linspace(min(X), max(X), 1000)
MatAbs = [Abs**i for i in range(n)]
MatAbs = transpose(MatAbs)
Yreg = multiplie(MatAbs, transpose(curve_coefficient))

# Compute error
error = multiplie(transpose(multiplie(A, transpose(curve_coefficient)) -
        transpose(Y)), (multiplie(A, transpose(curve_coefficient)) - transpose(Y)))

print("LSE :")
print("Fitting Line : ")
for i in range(len(curve_coefficient)):
    if i != 0:
        print('+', end=' ')
    print("{:.2e}*X^{:d}".format(curve_coefficient[i],i), end=' ')
print("\nTotal Error : ", error[0][0])


# Newton's Method
# In this case it is just the same computation
# This is because the function we want to minimize is quadratic. (square of norm of AX-b)
# Therefore, Newton' method for optimization find the minimum in only one iteration.
# Because the derivative is linear and Newtons finds the 0 of this linear function
# in one iteration.
# We would just compute the same function.
#  
print("Newton :")
print("Fitting Line : ")
for i in range(len(curve_coefficient)):
    if i != 0:
        print('+', end=' ')
    print(curve_coefficient[i], "*X^", i, end=' ')
print("\nNewton Total Error : ", error[0][0])


# Plot parameters
fig, ax = plt.subplots()
ax.grid()
plt.plot(X, Y, 'ro', label='Data points')
plt.plot(Abs, Yreg, label='Regression line')
legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large')
legend.get_frame().set_facecolor('grey')
plt.title('Polynomial Regression',fontsize=20)
plt.show()