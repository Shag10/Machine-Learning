import tensorflow as tf
import keras
import numpy as np

data=keras.datasets.imdb

(train_txt, train_label), (test_txt, test_label)=data.load_data(num_words=90000)

w_index=data.get_word_index()
w_index={k:(v+3) for k,v in w_index.items()}
w_index["<PAD>"]=0
w_index["<START>"]=1
w_index["<UNK>"]=2
w_index["<UNUSED>"]=3

rev_w_index=dict([(value,key) for (key, value) in w_index.items()])

train_txt=keras.preprocessing.sequence.pad_sequences(train_txt,value=w_index["<PAD>"],padding="post",maxlen=250)
test_txt=keras.preprocessing.sequence.pad_sequences(test_txt,value=w_index["<PAD>"],padding="post",maxlen=250)

def decode(text):
    return " ".join([rev_w_index.get(i,"?") for i in text])

# Main Model
model=keras.Sequential()
model.add(keras.layers.Embedding(90000,20))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(20,activation="relu"))
model.add(keras.layers.Dense(1,activation="sigmoid"))

model.summary()

model.compile(optimizer="adam",loss="binary_crossentropy", metrics=["accuracy"])
x_val=train_txt[:10000]
x_tr=train_txt[10000:]

y_val=train_label[:10000]
y_tr=train_label[10000:]

fmodel=model.fit(x_tr,y_tr,epochs=45,batch_size=512,validation_data=(x_val,y_val),verbose=1)

res=model.evaluate(test_txt,train_label)
print(res)

review = test_txt[0]
predict = model.predict([review])                 // You can retrieve a comment and analyze whether it is good comment or bad.
print("Review: ")
print(decode(review))
print("Prediction: "+str(predict[0]))
print("Actual: "+str(test_label[0]))
print(res)

model.save("model.h5")
