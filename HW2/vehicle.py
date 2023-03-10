import numpy as np
import math
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


def generate_landmark(K):
    landmarks = []
    r = 1
    for i in range(K):
        theta = (2*np.pi*i) / K
        landmarks.append([r*math.cos(theta), r*math.sin(theta)])
    landmarks = np.array(landmarks)
    return landmarks

def generate_true_position():
    random.seed(1)
    theta = random.uniform(0, 2*np.pi)
    r = random.uniform(0, 1)
    sample = np.array([r*math.cos(theta), r*math.sin(theta)])
    return sample

def generate_ranges(sample, landmarks):
    ranges = []
    for i in range(landmarks.shape[0]):
        di = np.linalg.norm(sample-landmarks[i])
        ni = np.random.normal(0, 0.3)
        ri = di + ni
        while(ri <= 0 ):
            ni = np.random.normal(0, 0.3)
            ri = ri + ni
        ranges.append(ri)
    ranges = np.array(ranges)
    return ranges

def cal_MAP(samples, landmarks, ranges):
    estimation = []
    for i in tqdm(range(samples.shape[0])):
        es = np.power(samples[i][0], 2) / np.power(0.25, 2)
        es = es + (np.power(samples[i][1], 2) / np.power(0.25, 2))
        for j in range(landmarks.shape[0]):
            di = np.linalg.norm(samples[i] - landmarks[j])
            es = es + (np.power(di-ranges[j], 2) / (2* np.power(0.3, 2)) )
        estimation.append(es)
    estimation = np.array(estimation)
    return estimation


def plot_pos_vehicle(K):
    true_pos = generate_true_position()
    landmarks = generate_landmark(K)
    ranges = generate_ranges(true_pos, landmarks)

    xx1, xx2 = np.meshgrid(
        np.linspace(-2, 2, int(400)),
        np.linspace(-2, 2, int(400))
    )
    # print("xx1.shape", xx1.shape) # 400 x 400
    # print("xx1", xx1)
    # print("xx1.ravel()", xx1.ravel().shape) # 160000 x 1
    # print("xx2", xx2) # 400 x 400

    samples = np.c_[xx1.ravel(), xx2.ravel()]
    # print("samples.shape", samples.shape) # 160000 x 2
    # print("samples", samples[400:440,:])

    Z = cal_MAP(samples, landmarks, ranges)

    estimate_idx = np.argmin(Z)
    estimate_pos = samples[estimate_idx]

    dis = np.linalg.norm(estimate_pos- true_pos)
    print("distance between true postion and estimation:", dis)

    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.5)
    plt.colorbar()
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.scatter(landmarks[:, 0], landmarks[:, 1], color='blue', label='landmarks')
    plt.scatter(true_pos[0], true_pos[1], marker='+', color= 'green', label='true vehicle position')
    plt.scatter(estimate_pos[0], estimate_pos[1], color= 'red', marker='x', label='estimate position')

    plt.legend(loc='upper left')
    plt.title("MAP Objective Estimatiuon Contours When K=" + str(landmarks.shape[0]))
    plt.show()

