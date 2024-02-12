import math
import streamlit as st
import numpy as np
from tensorflow import keras
from keras import datasets, utils, layers, models, optimizers
from utils import get_model_summary, cache_model, cache_session

"""
# 🧠 Deep Learning
"""

with st.echo():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    NUM_CLASSES = 10
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    y_train = utils.to_categorical(y_train, NUM_CLASSES)
    y_test = utils.to_categorical(y_test, NUM_CLASSES)

def compile_model(model: keras.Model):
    import sys
    import os
    # At this time, the v2.11+ optimizer `tf.keras.optimizers.Adam` runs
    # slowly on M1/M2 Macs, please use the legacy Keras optimizer instead,
    # located at `tf.keras.optimizers.legacy.Adam`.
    if sys.platform == "darwin" and "arm" in os.uname().machine:
        print("Using legacy optimizer")
        opt = optimizers.legacy.Adam(learning_rate=0.0005)
    else:
        opt = optimizers.Adam(learning_rate=0.0005)
    
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                    metrics=["accuracy"])

# itch: This does not seem to work cause it caches the wrong function
# @st.cache_resource
@cache_model("cifar10_model_functional")
def compile_model_functional():
    input_layer = layers.Input(shape=(32, 32, 3))
    x = layers.Flatten()(input_layer)
    x = layers.Dense(units=200, activation="relu")(x)
    x = layers.Dense(units=150, activation="relu")(x)
    output_layer = layers.Dense(units=10, activation="softmax")(x)
    model = models.Model(input_layer, output_layer)

    compile_model(model)
    model.fit(x_train, y_train, batch_size=32, epochs=10, shuffle=True)

    return model

# @st.cache_resource This does not seem to work cause it caches the wrong function
@cache_model("cifar10_model_sequential")
def compile_model_sequential():    
    model = models.Sequential([
        layers.Flatten(input_shape=(32, 32, 3)),
        layers.Dense(200, activation='relu'),
        layers.Dense(150, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    compile_model(model)
    model.fit(x_train, y_train, batch_size=32, epochs=10, shuffle=True)
    
    return model

# Using a Convolutional Neural Network
@cache_model("cifar10_model_conv")
def compile_model_conv():
    input_layer = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same")(input_layer)
    x = layers.BatchNormalization()(x)
    x =layers.LeakyReLU()(x)

    x = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(rate=0.5)(x)

    output_layer = layers.Dense(units=10, activation="softmax")(x)
    model = models.Model(input_layer, output_layer)
    model.summary()

    compile_model(model)
    model.fit(
        x_train,
        y_train,
        batch_size=32,
        epochs=10,
        shuffle=True,
        validation_data=(x_test, y_test),
    )
    
    return model


model_choice = st.selectbox("Choose Model", ["Sequential", "Functional", "Convolutional"])
if model_choice == "Sequential":
    model = compile_model_sequential()
elif model_choice == "Functional":
    model = compile_model_functional()
elif model_choice == "Convolutional":
    model = compile_model_conv()

if st.checkbox("Show model summary"):
    df, summ_lines = get_model_summary(model)
    st.dataframe(df, hide_index=True)
    cols = st.columns(3)
    for i, line in enumerate(summ_lines):
        title, metric = line.split(":")
        val = metric.split("(")[0].strip()
        cols[i].metric(title.strip(), val)

@cache_session(f"model_output_{model_choice}")
def get_model_output(model: keras.Model):
    cross_categorical_entropy, accuracy = model.evaluate(x_test, y_test)
    preds = model.predict(x_test)

    return cross_categorical_entropy, accuracy, preds
    

cross_categorical_entropy, accuracy, preds = get_model_output(model)
space, col1, col2, space = st.columns([2, 3, 3, 2])
col1.metric("Cross Categorical Entropy", round(cross_categorical_entropy, 3))
col2.metric("Accuracy", f"{round(accuracy * 100, 1)}%")

CLASSES = np.array(["airplane", "automobile", "bird", "cat", "deer", "dog",
                    "frog", "horse", "ship", "truck"])
preds_single = CLASSES[np.argmax(preds, axis=-1)]
actual_single = CLASSES[np.argmax(y_test, axis=-1)]

col1, col2 = st.columns([8, 2])
n_to_show = col1.number_input("Number of Predictions to Show", value=10, min_value=1, max_value=100, label_visibility="collapsed")
if col2.button("Reshuffle") or 'random_seed' not in st.session_state:
    st.session_state.random_seed = np.random.randint(0, 100000)

np.random.seed(st.session_state.random_seed)
indices = np.random.choice(range(len(x_test)), n_to_show)

NUM_COLS_PER_ROW = 4
for i in range(math.ceil(n_to_show / NUM_COLS_PER_ROW)):
    cols = st.columns(NUM_COLS_PER_ROW)

    for j in range(NUM_COLS_PER_ROW):
        col_index = i * NUM_COLS_PER_ROW + j
        if col_index >= n_to_show:
            break
 
        idx = indices[col_index]
        cols[j].image(x_test[idx], use_column_width=True)
        cols[j].text(f"pred = {preds_single[idx]}\nact = {actual_single[idx]}")
