import jax.numpy as jnp
import numpy as np
import jax

def hey(x):
    return x + np.random.rand(*x.shape)

x = np.array([2, 3])

print("hey")

np.random.seed(0)
for _ in range(2):
    y = hey(x)
    print(y)

print("hey_jit")

np.random.seed(0)
hey_jit = jax.jit(hey)
for _ in range(2):
    y = hey_jit(x)
    print(y)

print("correct hey_jit")

def correct_hey_jit(x, r):
    return x + r

np.random.seed(0)
hey_jit = jax.jit(hey)
for _ in range(2):
    r = np.random.rand(*x.shape)
    y = correct_hey_jit(x, r)
    print(y)
