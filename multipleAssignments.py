point1 = [4, 6]
point2 = [-3, 4]
point3 = [2, -5]
point4 = [-1, -7]
points = [point1, point2, point3, point4]
print(points)

x, y = point1
print(f"Point 1 is x={x}, y={y}")

x, y = y, x
print(f"After reflecting across the line y=x: x={x}, y={y}")

xCoords, yCoords = zip(*points)
print(xCoords, yCoords)

for x, y in zip(xCoords, yCoords):
    print (f"({x:2},{y:2})")

for index, point in enumerate(points):
    print(f"Point #{index} is {point}")

input("Press <ENTER> to continue")

# scikit-learn 0.23.2
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.DESCR)
print(digits.images.shape)
# The -1 argument means 'use whatever works
# to get it down to only one more dimension'
digitsX = digits.images.reshape(len(digits.images), -1)
print(digitsX.shape)
