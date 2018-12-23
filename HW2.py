# HW2 - Finding camera intrinsic parameters from 3D coordinates and 2D Image Plane - Eren ATAS 1334129

# Import Libraries
import numpy as np

# World Coordinates
X = np.array([[75.1267, 25.5095, 50.5957],
              [69.9077, 89.0903, 95.9291],
              [54.7216, 13.8624, 14.9294],
              [25.7508, 84.0717, 25.4282],
              [81.4285, 24.3525, 92.9264],
              [34.9984, 19.6595, 25.1084],
              [61.6045, 47.3289, 35.1660],
              [83.0829, 58.5264, 54.9724],
              [91.7194, 28.5839, 75.7200],
              [75.3729, 38.0446, 56.7822],
              [7.5854, 5.3950, 53.0798],
              [77.9167, 93.4011, 12.9906],
              [56.8824, 46.9391, 1.1902],
              [33.7123, 16.2182, 79.4285],
              [31.1215, 52.8533, 16.5649],
              [60.1982, 26.2971, 65.4079],
              [68.9215, 74.8152, 45.0542],
              [8.3821, 22.8977, 91.3337],
              [15.2378, 82.5817, 53.8342],
              [99.6135, 7.8176, 44.2678]])

# Image Plane x and y points
x = np.array([[0.0847, 0.0129],
              [0.0245, 0.0035],
              [0.1846, 0.0405],
              [0.0191, 0.0064],
              [0.0646, 0.0020],
              [0.0638, 0.0092],
              [0.0450, 0.0109],
              [0.0431, 0.0100],
              [0.0782, 0.0086],
              [0.0572, 0.0093],
              [0.0461, -0.0049],
              [0.0287, 0.0142],
              [0.0533, 0.0211],
              [0.0451, -0.0032],
              [0.0302, 0.0087],
              [0.0567, 0.0040],
              [0.0307, 0.0089],
              [0.0293, -0.0050],
              [0.0176, 0.0029],
              [-1.2160, -0.215]])

x = 1.0e+04 * x

print('X: ', X)
print('\n')
print('x: ', x)

# Creating A matrix
A = np.zeros([1, 8])

for i in range(X.shape[0]):
    append_to_a = np.array([[x[i][0]*X[i][0],
                       x[i][0]*X[i][1],
                       x[i][0]*X[i][2],
                       x[i][0],
                       -1*x[i][1]*X[i][0],
                       -1*x[i][1]*X[i][1],
                       -1*x[i][1]*X[i][2],
                       -x[i][1]]])
    A = np.concatenate([A, append_to_a], axis=0)

A = np.delete(A, 0, 0)

print('A: ', A)

# Instead of using SVD function, I have used eig function of numpy.linalg to get eigenvalues and eigenvectors.
aTa = np.transpose(A) @ A
eigenval, eigenvec = np.linalg.eig(aTa)

print('Eigenvalues: ', eigenval, '\n')
print('Eigenvectors: ', eigenvec)

for i in range(eigenvec.shape[0]):
    print(i, 'th column of A @ eigenvec: ', A @ eigenvec[:, i], '\n')

print('Last column is the closest to 0 which is our v.')

v = eigenvec[:, 7]

# norm of first 3 values of v for finding gamma.
gamma = np.linalg.norm([v[0], v[1], v[2]])

print('Gamma: ', gamma)

# Divide v to gamma so that the norm of r1 will be 1.
v = v / gamma

r1 = [v[4], v[5], v[6]]
alpha = np.linalg.norm(r1)

r1 = r1 / alpha
print(np.linalg.norm(r1))

# r2
r2 = [v[0], v[1], v[2]]
print('r2: \n', r2, '\n')

# Since R is orthonormal

r3 = np.cross(r1, r2)
print('r3: \n', r3, '\n')

R = [r1, r2, r3]


print('Determinant of R: ', np.linalg.det(R)) # Supposed to be very close to 1.

T_x = v[7]/alpha
T_y = v[3]

b = x[:, 0]*(X@r3.conj().T)

AA = np.array([-x[:, 0], (X@r1.conj().T) + v[7] / alpha])

np.linalg.lstsq(AA.conj().T, b)


solving_for_Tz_fx= np.linalg.lstsq(AA.conj().T,b)

T_z = solving_for_Tz_fx[0][0]
f_x = solving_for_Tz_fx[0][1]

T = np.array([T_x, T_y, T_z])

K=np.array([[f_x, 0, 0],
            [0, f_x, 0],
            [0, 0, 1]]);

print('\n', 'R:', '\n', R)
print('\n', 'T:', '\n', T)
print('\n', 'Alpha:', '\n', alpha)
print('\n', 'f:', '\n', f_x)
print('\n', 'K:', '\n', K)
