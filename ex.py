from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageGrab
from PIL import Image as im
from tkinter import *

global image_data
class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.canvas_width = 280  # MNIST 해상도와 동일하게 조정
        self.canvas_height = 280  # MNIST 해상도와 동일하게 조정
        self.canvas = Canvas(root, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)
        self.save_button1 = Button(root, text="Evaluate", command=self.save_image)
        self.save_button1.pack()
        self.save_button2 = Button(root, text="Exit", command=quit)
        self.save_button2.pack()
        
    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-10, y-10, x+10, y+10, fill='black')  # 그림 그리는 크기를 더 크게 조정
        
    def save_image(self):
        global image_data
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas_width
        y1 = y + self.canvas_height
        image = ImageGrab.grab().crop((x, y, x1, y1)).resize((28, 28), im.ANTIALIAS)  # 이미지 크기를 MNIST 해상도로 조정
        image = image.convert('L')  # 그레이스케일로 변환
        image_data = np.array(image)
        image_data = 255 - image_data  # 흑백 반전
        image_data = image_data.astype('float32') / 255.0
        self.root.destroy ()
        # 이제 image_data를 사용하여 원하는 저장 방식에 맞게 처리하면 됩니다

class CNN   :
    count = 0  # 클래스 변수
 
    def __init__(self, width, height):
        self.width = width
        self.height = height
        Rectangle.count += 1
 
    # 인스턴스 메서드
    def calcArea(self):
        area = self.width * self.height
        return area
 
    # 정적 메서드
    @staticmethod
    def isSquare(rectWidth, rectHeight):
        return rectWidth == rectHeight   
 
    # 클래스 메서드
    @classmethod
    def printCount(cls):
        print(cls.count)   
        
class GuI_Main_System:
    def __init__(self):
        self.menu = {}
        self.orders = {}

    def add_menu_item(self, item_name, price):
        self.menu[item_name] = price
        print(f"Menu item '{item_name}' added with price {price}.")

    def place_order(self, customer_name, items):
        if customer_name not in self.orders:
            self.orders[customer_name] = {}
        
        order_total = 0
        for item_name in items:
            if item_name in self.menu:
                item_price = self.menu[item_name]
                self.orders[customer_name][item_name] = item_price
                order_total += item_price
        
        if order_total > 0:
            print(f"Order placed for customer '{customer_name}'. Total amount: {order_total}.")
        else:
            print("Invalid order. Please check the menu items.")

    def get_order_total(self, customer_name):
        if customer_name in self.orders:
            order_total = sum(self.orders[customer_name].values())
            print(f"Total amount for customer '{customer_name}': {order_total}.")
        else:
            print(f"No orders found for customer '{customer_name}'.")

(x_train, y_train), (x_test, y_test) = mnist.load_data() # mnist 데이타 받아오기(ai 학습의 기초 데이타)
print("x_train shape", x_train.shape)
print("y_train shape", y_train.shape)
print("x_test shape", x_test.shape)
print("y_test shape", y_test.shape)


print("----------------------------------------")
X_train = x_train.reshape(60000, 784)
X_test = x_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print("X Training matrix shape", X_train.shape)
print("X Testing matrix shape", X_test.shape)
print("----------------------------------------")
Y_train = to_categorical(y_train, 10)
Y_test = to_categorical(y_test, 10)
print("Y Training matrix shape", Y_train.shape)
print("Y Testing matrix shape", Y_test.shape)

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=128, epochs=10, verbose=1)

score = model.evaluate(X_test, Y_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])

predicted_classes = np.argmax(model.predict(X_test), axis=1)
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]

print(X_test[0])
plt.figure()
for i in range(9):
    plt.subplot(3,3,i+1)
    correct = correct_indices[i]
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))
plt.tight_layout()
plt.show()
while(TRUE):
    root = Tk()
    app = DrawingApp(root)
    root.mainloop()
    prediction = model.predict(image_data.flatten().reshape(1,784))
    predicted_label = np.argmax(prediction)
    print(f"Predicted label: {predicted_label}")
    plt.figure()
    plt.imshow(image_data, cmap='gray')
    plt.title("Predicted {}".format(predicted_label))
    plt.tight_layout()
    plt.show()