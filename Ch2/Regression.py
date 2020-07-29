import numpy as np
from matplotlib import pyplot as plt

def gen_test_data():
    """
    生成数据
    """
    x = np.random.uniform(-10.0, 10., 100)
    eps = np.random.normal(0., 3, 100)
    y = 1.477 * x + 0.089 + eps
    return x, y

def mse(b, w, points):
    """
    误差函数
    """
    totalError = sum((y - w*x - b)**2 for x,y in points)
    return totalError / len(points)

def step_gradient(b_current, w_current, points, lr):
    """
    计算梯度
    """
    b_gradient = 0
    w_gradient = 0
    N = len(points)
    for x, y in points:
        b_gradient += (2/N)*(w_current * x + b_current - y)
        w_gradient += (2/N)* x *(w_current *x + b_current - y)
    new_b = b_current - (lr * b_gradient)
    new_w = w_current - (lr * w_gradient)
    return new_b, new_w


def gradient_descent(points, starting_b, starting_w, lr, num_iterations):
    """
    梯度下降
    """
    b = starting_b
    w = starting_w
    for step in range(num_iterations):
        b, w = step_gradient(b, w, points, lr)
        if step % 10 == 0:
            print(f"iteration:{step}, loss:{mse(b, w, points)}, w:{w}, b:{b}")
    return b, w

def main():
    x, y = gen_test_data()
    points = np.array(list(zip(x, y)))
    b, w = gradient_descent(points, 0., 0., 0.01, 1000)

    plt.scatter(x, y, c="r")
    plt.plot(x, w*x + b, "b", lw=2)
    plt.show()

if __name__ == "__main__":
    main()