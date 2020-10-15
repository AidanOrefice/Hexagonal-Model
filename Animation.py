import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def main():
    array = np.zeros((16))
    array[0] = 1
    fig, ax = plt.subplots()
    global line1
    global line2
    x, y = [], []
    x1, y1 = [index_to_xy(0,4)[0]], [index_to_xy(0,4)[0]]
    for i in range(1,len(array)):
        xy = (index_to_xy(i,4))
        x.append(xy[0])
        y.append(xy[1])
    global F
    F = []
    for i in range(2,16):
        F.append(list(array))
        array[i-1] = 0
        array[i] = 1 
    line1 = ax.plot(x, y, color='green', marker = 'o', ls = '', markersize = 25)[0]
    line2 = ax.plot(x1, y1, color='blue', marker = 'o', ls = '', markersize = 25)[0]
    anim = FuncAnimation(fig, animate, interval=100, frames=15)
    plt.draw()
    plt.show()

def index_to_xy(index, width):
    row = int(str(index / width)[0])
    y = row
    if_even = row % 2 == 0
    if if_even:
        x = index - (row * width)
    else:
        x = index - (row * width) + 0.5
    return (x,y)

def animate(i):
    global line1
    global line2
    global F
    x = F[i]
    indices1 = [k for k, j in enumerate(x) if j == 0]
    indices2 = [k for k, j in enumerate(x) if j == 1]
    x1, y1 = [], []
    x2, y2 = [], []
    for j in indices1:
        xy = index_to_xy(j, 4)
        x1.append(xy[0])
        y1.append(xy[1])
    for j in indices2:
        xy = index_to_xy(j, 4)
        x2.append(xy[0])
        y2.append(xy[1])
    print(x1, x2)
    print(y1, y2)
    line1.set_ydata(y1)
    line2.set_ydata(y2)

if __name__ == "__main__":
    main()