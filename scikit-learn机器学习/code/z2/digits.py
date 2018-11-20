from sklearn import datasets
from matplotlib import pyplot as plt

# 手写数字数据集
digits = datasets.load_digits()  # <class 'sklearn.utils.Bunch'>

# 将数据集中的手写数字图像和标签打包成元组,这里得到[(图像,标签),...]
images_and_labels = list(zip(digits.images, digits.target))

plt.figure(figsize=(8, 6), dpi=200)
# 遍历(image=图像,label=标签),i=遍历序号.只遍历前8个
for i, (image, label) in enumerate(images_and_labels[:8]):
    # 在子图上绘制出来看一下
    plt.subplot(2, 4, i + 1)  # 激活对应位置的子图
    plt.axis('off')  # 不显示坐标
    plt.imshow(image, cmap='gray_r', interpolation='nearest')  # interpolation='nearest'设置最近邻插值
    plt.title("Digit:%i" % label, fontsize=20)
plt.show()

print("原始图像的shape:{0}".format(digits.images.shape))  # (1797, 8, 8)
print("对应的数据的shape:{0}".format(digits.data.shape))  # (1797, 64)
