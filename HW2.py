#!/usr/bin/env python
# coding: utf-8

"""
HW2: Linear Discriminant Analysis
In hw2, you need to implement Fisher’s linear discriminant by using only numpy,
then train your implemented model by the provided dataset
and test the performance with testing data

Please note that only **NUMPY** can be used to implement your model,
you will get no points by simply calling
sklearn.discriminant_analysis.LinearDiscriminantAnalysis

Ref.: https://github.com/sthalles/fishers-linear-discriminant
"""
import pandas as pd
import numpy as np
from numpy.linalg import pinv
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

ROOT = ''


def load_data():
    x_train = pd.read_csv(ROOT + "x_train.csv").values
    y_train = pd.read_csv(ROOT + "y_train.csv").values[:, 0]
    x_test = pd.read_csv(ROOT + "x_test.csv").values
    y_test = pd.read_csv(ROOT + "y_test.csv").values[:, 0]

    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = load_data()
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

# 1. Compute the mean vectors mi, (i=1,2) of each 2 classes
# y = 0,1
x1 = x_train[y_train == 0]
m1 = np.mean(x1, axis=0)
x2 = x_train[y_train == 1]
m2 = np.mean(x2, axis=0)

assert m1.shape == (2,)
assert m2.shape == (2,)
print(f'Mean vector of class 1: {m1}\nMean vector of class 2: {m2}\n')


# 2. Compute the Within-class scatter matrix SW
# Ref. CH3 P.28
diff = x1 - m1
sw1 = diff.T @ diff
diff = x2 - m2
sw2 = diff.T @ diff
sw = sw1 + sw2

assert sw.shape == (2, 2)
print(f"Within-class scatter matrix SW:\n{sw}\n")

# 3.  Compute the Between-class scatter matrix SB

diff = m2 - m1
# Final shape = 2, 2 => reshape
diff = diff.reshape(2, 1)
sb = diff @ diff.T

assert sb.shape == (2, 2)
print(f"Between-class scatter matrix SB:\n{sb}\n")

# 4. Compute the Fisher’s linear discriminant (optimized w)
# w ∝ (sw)^(-1) x (m2 - m1)
inv_SW = pinv(sw)
diff = m2 - m1
w = inv_SW @ np.expand_dims(diff, 1)
assert w.shape == (2, 1)
print(f" Fisher’s linear discriminant:\n{w}\n")

# 5. Project the test data by linear discriminant to get the class prediction
#    by nearest-neighbor rule and calculate the accuracy score
# you can use accuracy_score function from sklearn.metric.accuracy_score

x_test_dot = x_test @ w

tot = 0
for m in [m1, m2]:
    tot += np.dot(np.squeeze(w), m)

w0 = 0.5 * tot
print(f"Optimal threshold: {w0}\n")


predictions = [0 if pred < w0 else 1 for pred in x_test_dot]

# Another NN rule by cal. testing and training nearest point's label
# Acc. : 0.88
# test_size = x_test.shape[0]
# predictions = np.zeros(test_size)
# for i, xtp in enumerate(x_test_proj):
#     min_dis = w0
#     for j, xp in enumerate(x_train_proj):
#         d0 = (xtp[0]-xp[0]) ** 2
#         d1 = (xtp[1]-xp[1]) ** 2
#         d = np.sqrt(d0 + d1)
#         if d < min_dis:
#             pred = y_train[j]
#             min_dis = d
#     predictions[i] = pred

acc = accuracy_score(y_test, predictions)

print(f"Accuracy of test-set: {acc}")


# # draw the picture
# Reshape for np.dot of w because point is in shape(2,)
w = w.reshape(2,)

fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#C9306B', '#278AA8']
labels = ['Class 1', 'Class 2']
max = float('-inf')
min = float('inf')

for point, pred in zip(x_train, y_train):
    # Draw points of origin and projection point
    ax.scatter(point[0], point[1], color=colors[pred], alpha=0.15, s=15)
    proj = np.dot(point, w)/np.dot(w, w) * w
    scatter = ax.scatter(proj[0], proj[1], color=colors[pred], s=15)
    # Draw lines between origin and projection point
    plt.plot([point[0], proj[0]], [point[1], proj[1]],
             lw=0.5, alpha=0.1, c='#327185')
    # Cal. projection point x value max and min
    max = proj[0] if proj[0] > max else max
    min = proj[0] if proj[0] < min else min

# Show the label of two classes
pop_a = mpatches.Patch(color=colors[0], label=labels[0])
pop_b = mpatches.Patch(color=colors[1], label=labels[1])
plt.legend(handles=[pop_a, pop_b])

# y = a x + b
slope = w[1]/w[0]  # cal. slope y/x
x = [min, max]
y = [slope*min, slope*max]
# project line
plt.plot(x, y, lw=1, c='#327185')

# plot the mean point of each class
ax.scatter(m1[0], m1[1], color='#327185', s=100, marker="X")
ax.scatter(m2[0], m2[1], color='#327185', s=100, marker="X")

# Draw mean point of each class's  line
line = mlines.Line2D([m1[0], m2[0]], [m1[1], m2[1]], color='#3C3147')
ax.add_line(line)

# Show info
plt.title('Projection Line: w=%f, b=%f' % (slope, 0))
plt.savefig('Projection.png', dpi=300)
plt.show()
