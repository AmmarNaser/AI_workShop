import os


def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

output_dir = "/Users/apple/Documents/WS/ai_401/myenv/CNN/fashion-mnist/data/fashion/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

convert(os.path.join(output_dir, "train-images-idx3-ubyte"),
        os.path.join(output_dir, "train-labels-idx1-ubyte"),
        os.path.join(output_dir, "mnist_train.csv"), 60000)

convert(os.path.join(output_dir, "t10k-images-idx3-ubyte"),
        os.path.join(output_dir, "t10k-labels-idx1-ubyte"),
        os.path.join(output_dir, "mnist_test.csv"), 10000)


# convert("/Users/apple/Documents/WS/ai_401/myenv/CNN/fashion-mnist/data/fashion/train-images-idx3-ubyte",
#         "/Users/apple/Documents/WS/ai_401/myenv/CNN/fashion-mnist/data/fashion/train-labels-idx1-ubyte",
#         "mnist_train.csv", 60000)

# convert("/Users/apple/Documents/WS/ai_401/myenv/CNN/fashion-mnist/data/fashion/t10k-images-idx3-ubyte",
#         "/Users/apple/Documents/WS/ai_401/myenv/CNN/fashion-mnist/data/fashion/t10k-labels-idx1-ubyte",
#         "mnist_test.csv", 10000)


