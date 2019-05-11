import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

colors = [np.array([0, 0, 0]), np.array([255, 255, 255]), [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255],
          [255, 0, 255]]


def seg_kmeans_gray():
    # 读取图片
    img = cv.imread('MRI-4.jpg', cv.IMREAD_GRAYSCALE)
    # 展平
    img_flat = img.reshape((img.shape[0] * img.shape[1], 1))
    img_flat = np.float32(img_flat)
    print(img_flat)

    # 迭代参数
    criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_MAX_ITER, 20, 0.5)
    flags = cv.KMEANS_RANDOM_CENTERS

    # 进行聚类
    compactness, labels, centers = cv.kmeans(img_flat, 4, None, criteria, 10, flags)

    # 显示结果
    img_output = labels.reshape((img.shape[0], img.shape[1]))
    plt.subplot(121), plt.imshow(img, 'gray'), plt.title('input')
    plt.subplot(122), plt.imshow(img_output, 'gray'), plt.title('kmeans')
    plt.show()


def seg_kmeans_color():
    img = cv.imread('Bird-2.jpg', cv.IMREAD_COLOR)
    # 变换一下图像通道bgr->rgb，否则很别扭啊
    b, g, r = cv.split(img)
    img = cv.merge([r, g, b])

    # 3个通道展平
    img_flat = img.reshape((img.shape[0] * img.shape[1], 3))
    img_flat = np.float32(img_flat)

    # 迭代参数
    criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_MAX_ITER, 20, 0.5)
    flags = cv.KMEANS_RANDOM_CENTERS

    # 聚类,这里k=2
    compactness, labels, centers = cv.kmeans(img_flat, 2, None, criteria, 10, flags)

    # 显示结果
    img_output = labels.reshape((img.shape[0], img.shape[1]))
    plt.subplot(121), plt.imshow(img), plt.title('input')
    plt.subplot(122), plt.imshow(img_output, 'gray'), plt.title('kmeans')
    plt.show()

def Get_Feature(arr):
    feture = []
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            feture.append([arr[i][j][0], arr[i][j][1], arr[i][j][2]])
    return feture

def Calc_Dist(a, b):
    return math.sqrt(math.pow(a[0]-b[0],2) + math.pow(a[1]-b[1],2)+ math.pow(a[2]-b[2],2))

def Centres_Update(lst, centres, k):
    categories = [[] for i in range(k)]
    for each in lst:
        min = 255 ** 3
        key = 0
        for i in range(k):
            tmp = Calc_Dist(each, centres[i])
            if min > tmp:
                min = tmp
                key = i
        categories[key].append(each)
    for i in range(k):
        centres[i] = np.mean(categories[i], axis=0)
    return centres


def kmeans(name, k):
    img = cv.imread(name)
    b, g, r = cv.split(img)
    img = cv.merge([r, g, b])
    img_arr = np.array(img)
    Feature = Get_Feature(img_arr)
    randoms = [int(np.random.random() * img_arr.shape[0]) for i in range(k)]
    centres = [Feature[randoms[i]] for i in range(k)]
    cou = 0
    while True:
        tmp_centres = [tmp for tmp in centres]
        centres = Centres_Update(Feature, centres, k)
        s = 0
        for x in range(k):
            s += Calc_Dist(tmp_centres[x], centres[x])
        cou += 1
        print("当前差值：{}".format(s))
        print("已经完成第{}次迭代".format(cou))
        if s < 0.5:
            break

    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            min = 255 ** 3
            key = 0
            for x in range(k):
                tmp = Calc_Dist(img_arr[i][j], centres[x])
                if min > tmp:
                    min = tmp
                    key = x
            for x in range(3):
                #img_arr[i][j][x] = colors[key][x]
                img_arr[i][j][x] = centres[key][x]

    win = cv.namedWindow('kmeans', flags=0)
    cv.imshow('kmeans', img_arr)
    cv.waitKey(0)



if __name__ == '__main__':
    # seg_kmeans_gray()
    # seg_kmeans_color()
    #kmeans('Bird-2.jpg', 2)
    kmeans('Scenery-3.jpg', 4)
    #kmeans('MRI-4.jpg', 4)
