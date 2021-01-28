#!/usr/bin/env python

import sys
import math
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# set seed
np.random.seed(7)

# import data set
df = pd.read_csv('./data/passengers.csv', sep=';', parse_dates=True, index_col=0)

data = df.values

# using keras often requires the data type float32
data = data.astype('float32')

# for split all data to train and test
X_train, y_train, X_test, y_test = [], [], [], []
# for split train to train and test
X_train2, y_train2, X_test2, y_test2 = [], [], [], []


class Features:
    @staticmethod
    def autocorrelation(x, n=1, t=1):
        x = np.reshape(x, len(x)*n)
        temp = np.corrcoef(np.array([x[0:len(x)-t], x[t:len(x)]]))
        temp = temp.tolist()
        return temp[0][1]

    def test_autocorrelation(self):
        train_input = y_train.reshape((-1, 1))
        print(self.autocorrelation(train_input))

    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    @staticmethod
    def neurel_network(num_layer=4, input_dim=1, output_dim=1):
        print("\n*** Creating a new model with #{} hidden layers. ***\n".format(num_layer))
        mdl0 = Sequential()
        mdl0.add(Dense(num_layer, input_dim=input_dim, activation='relu'))
        mdl0.add(Dense(output_dim))
        mdl0.compile(loss='mean_squared_error', optimizer='adam')
        return mdl0

    @staticmethod
    def split_sequence(data_, n_steps_in=1, n_steps_out=1, skip=0):
        x, y = list(), list()
        for i in range(len(data_)):
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out + skip
            if out_end_ix > len(data_):
                break
            seq_x, seq_y = data_[i:end_ix, 0], data_[end_ix + skip:out_end_ix, 0]
            x.append(seq_x)
            y.append(seq_y)
        return np.array(x), np.array(y)

    def data_parser(self, n_steps_in=1, n_steps_out=1, skip=0):
        global data
        print("Parsing data with the percent of {}".format(window.ratio))
        window.x = int((len(data)/100)*window.ratio)

        train_file = data[0:window.x, :]
        t = pd.DataFrame(train_file)
        t.to_csv("./data/train.csv")
        dt = pd.read_csv('./data/train.csv', sep=',', index_col=0)
        dat = dt.values
        dat = dat.astype('float32')
        train = dat[0:, :]

        test_file = data[window.x:, :]
        t = pd.DataFrame(test_file)
        t.to_csv("./data/test.csv")
        dt = pd.read_csv('./data/test.csv', sep=',', index_col=0)
        dat = dt.values
        dat = dat.astype('float32')
        test = dat[0:, :]

        global X_train, y_train, X_test, y_test
        X_train, y_train = self.split_sequence(train, n_steps_in, n_steps_out, skip)
        X_test, y_test = self.split_sequence(test, n_steps_in, n_steps_out, skip)

    def train_data_parser(self, n_steps_in=1, n_steps_out=1, skip=0):
        global data
        train_ratio = 80
        window.x = int((len(data)/100)*window.ratio)
        window.x2 = int((window.x/100)*train_ratio)

        train_file = data[0:window.x2, :]
        t = pd.DataFrame(train_file)
        t.to_csv("./data/train_optimize.csv")
        dt = pd.read_csv('./data/train_optimize.csv', sep=',', index_col=0)
        dat = dt.values
        dat = dat.astype('float32')
        train = dat[0:, :]

        test_file = data[window.x2:window.x, :]
        t = pd.DataFrame(test_file)
        t.to_csv("./data/test_optimize.csv")
        dt = pd.read_csv('./data/test_optimize.csv', sep=',', index_col=0)
        dat = dt.values
        dat = dat.astype('float32')
        test = dat[0:, :]

        global X_train2, y_train2, X_test2, y_test2
        X_train2, y_train2 = self.split_sequence(train, n_steps_in, n_steps_out, skip)
        X_test2, y_test2 = self.split_sequence(test, n_steps_in, n_steps_out, skip)

    @staticmethod
    def split_sequence2(data_, lags, n_steps_in=3, n_steps_out=1):
        x, y = list(), list()
        last_element = lags[n_steps_in - 1]
        for i in range(len(data_)):
            end_ix = i + last_element
            out_end_ix = end_ix + n_steps_out
            if out_end_ix > len(data_):
                break
            for k in lags:
                x.append(data_[i+(k-1), 0])
            y.append(data_[last_element+i, 0])
        return np.array(x), np.array(y)

    def data_parser2(self, lags, train_ratio=85, n_steps_in=3, n_steps_out=1):
        global data
        print("Parsing data with the percent of {}".format(train_ratio))
        window.x = int((len(data)/100)*train_ratio)

        train_file = data[0:window.x, :]
        t = pd.DataFrame(train_file)
        t.to_csv("./data/train.csv")
        dt = pd.read_csv('./data/train.csv', sep=',', index_col=0)
        dat = dt.values
        dat = dat.astype('float32')
        train = dat[0:, :]

        test_file = data[window.x:, :]
        t = pd.DataFrame(test_file)
        t.to_csv("./data/test.csv")
        dt = pd.read_csv('./data/test.csv', sep=',', index_col=0)
        dat = dt.values
        dat = dat.astype('float32')
        test = dat[0:, :]

        global X_train, y_train, X_test, y_test
        X_train, y_train = self.split_sequence2(train, lags, n_steps_in, n_steps_out)
        X_test, y_test = self.split_sequence2(test, lags, n_steps_in, n_steps_out)
        X_train = X_train.reshape((-1, n_steps_in))
        y_train = y_train.reshape((-1, n_steps_out))
        X_test = X_test.reshape((-1, n_steps_in))
        y_test = y_test.reshape((-1, n_steps_out))

    def train_data_parser2(self, lags, train_ratio=80, n_steps_in=1, n_steps_out=1):
        global data
        window.x = int((len(data)/100)*window.ratio)
        window.x2 = int((window.x/100)*train_ratio)

        train_file = data[0:window.x2, :]
        t = pd.DataFrame(train_file)
        t.to_csv("./data/train_optimize.csv")
        dt = pd.read_csv('./data/train_optimize.csv', sep=',', index_col=0)
        dat = dt.values
        dat = dat.astype('float32')
        train = dat[0:, :]

        test_file = data[window.x2:window.x, :]
        t = pd.DataFrame(test_file)
        t.to_csv("./data/test_optimize.csv")
        dt = pd.read_csv('./data/test_optimize.csv', sep=',', index_col=0)
        dat = dt.values
        dat = dat.astype('float32')
        test = dat[0:, :]

        global X_train2, y_train2, X_test2, y_test2
        X_train2, y_train2 = self.split_sequence2(train, lags, n_steps_in, n_steps_out)
        X_test2, y_test2 = self.split_sequence2(test, lags, n_steps_in, n_steps_out)
        X_train2 = X_train2.reshape((-1, n_steps_in))
        y_train2 = y_train2.reshape((-1, n_steps_out))
        X_test2 = X_test2.reshape((-1, n_steps_in))
        y_test2 = y_test2.reshape((-1, n_steps_out))

    def best_number_of_layer(self, case, input_dim=1, output_dim=1):
        global X_train2, y_train2, X_test2, y_test2
        mdl1 = []
        rmse_scores = []

        x = str(window.lineEdit_0.text())
        y = str(window.lineEdit_1.text())

        if x is "" or y is "":
            return -1

        try:
            x = int(x)
            y = int(y)

        except ValueError:
            return -1

        if x > y:
            x, y = y, x

        if x is y:
            return -1

        self.train_data_parser(80, input_dim, output_dim)

        if case is 0:
            for i in range(0, (y-x+1)):

                window.progress.setValue(int(100/(y-x+1))*(i+1)-10)
                print("Testing with #", i+x)
                mdl1.append(self.neurel_network(i+x, input_dim, output_dim))
                mdl1[i].fit(X_train2, y_train2, epochs=200, batch_size=2, verbose=0)
                test_predict = mdl1[i].predict(X_test2)
                mse = ((y_test2.reshape(-1, 1) - test_predict.reshape(-1, 1)) ** 2).mean()
                rmse_scores.append(math.sqrt(mse))
                print("Mean Squared Error: {}".format(mse))
                print("\n")

            return rmse_scores.index(min(rmse_scores)) + x, min(rmse_scores)

        elif case is 1:
            for i in range(0, (y-x+1)):

                window.progress.setValue(int(100/(y-x+1))*(i+1)-10)
                print("Testing with #", i+x)
                mdl1.append(self.neurel_network(i+x, input_dim, output_dim))
                mdl1[i].fit(X_train2, y_train2, epochs=200, batch_size=2, verbose=0)
                test_predict = mdl1[i].predict(X_test2)
                mse = ((y_test2.reshape(-1, 1) - test_predict.reshape(-1, 1)) ** 2).mean()
                rmse_scores.append(math.sqrt(mse))
                print("Mean Squared Error: {}".format(mse))
                print("\n")

            return rmse_scores.index(min(rmse_scores)) + x, min(rmse_scores)

        elif case is 2:
            for i in range(0, (y - x + 1)):
                window.progress.setValue(int(100 / (y - x + 1)) * (i + 1) - 10)
                print("Testing with #", i + x)
                mdl1.append(self.neurel_network(i + x, window.n_step + 1, window.n_step))
                mdl1[i].fit(X_train2, y_train2, epochs=200, batch_size=2, verbose=0)
                test_predict = mdl1[i].predict(X_test2)
                mse = ((y_test2.reshape(-1, 1) - test_predict.reshape(-1, 1)) ** 2).mean()
                rmse_scores.append(math.sqrt(mse))
                print("Mean Squared Error: {}".format(mse))
                print("\n")

            return rmse_scores.index(min(rmse_scores)) + x, min(rmse_scores)

        elif case is 3:
            if len(window.lags) is 1:
                self.train_data_parser(80, window.lags[0], 1)

                for i in range(0, (y - x + 1)):
                    window.progress.setValue(int(100 / (y - x + 1)) * (i + 1) - 10)
                    print("Testing with #", i + x)
                    mdl1.append(self.neurel_network(i + x, window.lags[0], 1))
                    mdl1[i].fit(X_train2, y_train2, epochs=200, batch_size=2, verbose=0)
                    test_predict = mdl1[i].predict(X_test2)
                    mse = ((y_test2.reshape(-1, 1) - test_predict.reshape(-1, 1)) ** 2).mean()
                    rmse_scores.append(math.sqrt(mse))
                    print("Mean Squared Error: {}".format(mse))
                    print("\n")

                return rmse_scores.index(min(rmse_scores)) + x, min(rmse_scores)

            else:
                window.lags.sort()
                self.train_data_parser2(window.lags, 80, len(window.lags), 1)

                for i in range(0, (y - x + 1)):
                    window.progress.setValue(int(100 / (y - x + 1)) * (i + 1) - 10)
                    print("Testing with #", i + x)
                    mdl1.append(self.neurel_network(window.best_neuron_number, len(window.lags), 1))
                    mdl1[i].fit(X_train2, y_train2, epochs=200, batch_size=2, verbose=0)
                    test_predict = mdl1[i].predict(X_test2)
                    mse = ((y_test2.reshape(-1, 1) - test_predict.reshape(-1, 1)) ** 2).mean()
                    rmse_scores.append(math.sqrt(mse))
                    print("Mean Squared Error: {}".format(mse))
                    print("\n")

                return rmse_scores.index(min(rmse_scores)) + x, min(rmse_scores)

    def prediction(self, case=0, input_dim=1, output_dim=1):
        global data, X_train, y_train, X_test, y_test

        dt = pd.read_csv('./data/train.csv', sep=',', index_col=0)
        dat = dt.values
        dat = dat.astype('float32')
        train = dat[0:, :]

        dt = pd.read_csv('./data/test.csv', sep=',', index_col=0)
        dat = dt.values
        dat = dat.astype('float32')
        test = dat[0:, :]

        x_input = np.array([], dtype='float32')
        prediction = np.array([], dtype='float32')
        prediction_expected = np.array([], dtype='float32')
        mode = 2

        if window.is_lag and len(window.lags) is 1:
            mode = 0
            input = window.lags[0]
        elif window.is_lag:
            mode = 1
            input = len(window.lags)
        else:
            mode = 2
            input = 1

        if case is 0:  # prediction
            mdl = self.neurel_network(window.best_neuron_number, input_dim, output_dim)
            mdl.fit(X_train, y_train, epochs=200, batch_size=2, verbose=0)

            prediction = mdl.predict(X_test, input_dim, output_dim)
            prediction_expected = y_test
            window.progress.setValue(100)

        elif case is 1:  # direct multi step forecasting
            for i in range(window.n_step):
                if mode is 0:
                    feature.data_parser(input, 1, i)
                    x_input = np.append(prediction_expected, train[-input:, 0])
                elif mode is 1:
                    print()
                else:
                    feature.data_parser(input, 1, i)
                    x_input = np.append(prediction_expected, train[-1, 0])
                x_input = x_input.reshape((-1, input))
                mdl = self.neurel_network(window.best_neuron_number, input, 1)
                mdl.fit(X_train, y_train, epochs=200, batch_size=2, verbose=0)

                prediction = np.append(prediction, mdl.predict(x_input))

            prediction = prediction.reshape((-1, 1))
            prediction_expected = np.append(prediction_expected, test[0:window.n_step, 0])

        elif case is 2:  # multiple output forecasting
            if input is 1:
                input = 12
            mdl = self.neurel_network(window.best_neuron_number, input, window.n_step)
            mdl.fit(X_train, y_train, epochs=200, batch_size=2, verbose=0)

            x_input = np.append(prediction_expected, train[-input:, 0])
            x_input = x_input.reshape((1, input))

            prediction = mdl.predict(x_input, input_dim)
            prediction_expected = np.append(prediction_expected, test[0:window.n_step, 0])

        elif case is 3:  # recursive n step forecasting
            mdl = self.neurel_network(window.best_neuron_number, window.n_step, 1)
            mdl.fit(X_train, y_train, epochs=200, batch_size=2, verbose=0)

            for i in range(window.n_step):

                x_input = []
                x_input = np.append(x_input, train[len(train)-window.n_step+i:len(train), 0])
                x_input = np.append(x_input, prediction)
                x_input = x_input.reshape((-1, window.n_step))

                print("x_input")
                print(x_input)

                prediction = np.append(prediction, mdl.predict(x_input, 1, 1))
            print("prediction")
            print(prediction)
            prediction = prediction.reshape((-1, 1))
            prediction_expected = np.append(prediction_expected, test[0:window.n_step, 0])

        window.progress.setValue(100)

        # plot baseline and predictions
        ax = window.fig.add_subplot(111)
        ax.clear()
        mape = feature.mean_absolute_percentage_error(prediction_expected, prediction)
        mse = ((prediction_expected.reshape(-1, 1) - prediction.reshape(-1, 1)) ** 2).mean()
        ax.set_title('Prediction quality: {:.2f} MSE ({:.2f} RMSE) ({:.2f} MAPE)'.format(mse, math.sqrt(mse), mape))
        ax.plot(prediction_expected.reshape(-1, 1), label='Observed', color='#006699')
        ax.plot(prediction.reshape(-1, 1), label='Prediction', color='#ff0066')
        ax.legend(loc='best')
        ax.grid()

        window.canvas.draw()
        window.thread_flag = 1


feature = Features()


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setGeometry(600, 200, 1200, 600)
        self.setWindowTitle("Time Series Forecasting with MLP Tool")

        self.cb = QCheckBox("Do you want to optimize\nneuron number?", self)
        self.cb.stateChanged.connect(self.click_box)
        self.cb.move(20, 20)
        self.cb.resize(420, 40)

        self.lineEdit_0 = QLineEdit()
        self.lineEdit_1 = QLineEdit()
        self.lineEdit_2 = QLineEdit()
        self.lineEdit_3 = QLineEdit()
        self.lineEdit_4 = QLineEdit()
        self.lineEdit_5 = QLineEdit()

        self.label_0 = QLabel()
        self.label_2 = QLabel()
        self.label_3 = QLabel()
        self.label_4 = QLabel()
        self.label_6 = QLabel()
        self.label_7 = QLabel()
        self.label_8 = QLabel()
        self.label_9 = QLabel()

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setFocusPolicy(Qt.StrongFocus)
        self.slider.setTickPosition(QSlider.TicksBothSides)
        self.slider.setTickInterval(10)
        self.slider.setSingleStep(1)
        self.slider.setValue(85)

        self.pushButton0 = QPushButton("Prediction")
        self.pushButton0.clicked.connect(self.basic_predict_button)
        self.pushButton1 = QPushButton("Direct Multi-step\nForecast")
        self.pushButton1.clicked.connect(self.n_step_forecast_button)
        self.pushButton2 = QPushButton("Multiple Output\nForecast")
        self.pushButton2.clicked.connect(self.multi_output_forecast_button)
        self.pushButton3 = QPushButton("Recursive Multi-step\nForecast")
        self.pushButton3.clicked.connect(self.recursive_forecast_button)

        self.progress = QProgressBar(self)
        self.progress.setMaximum(100)

        self.w = None
        self.is_lag = 0
        self.n_step = 0
        self.lags = []
        self.optimize_state = 0
        self.x = 0
        self.x2 = 0
        self.ratio = 85
        self.best_neuron_number = 12
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.canvas)
        left_layout.addWidget(self.progress)

        mid_layout = QVBoxLayout()
        mid_layout.addWidget(self.label_0)
        mid_layout.addWidget(self.slider)
        mid_layout.addWidget(self.label_9)
        mid_layout.addWidget(self.lineEdit_5)
        mid_layout.addWidget(self.cb)
        mid_layout.addWidget(self.label_3)
        mid_layout.addWidget(self.lineEdit_0)
        mid_layout.addWidget(self.label_4)
        mid_layout.addWidget(self.lineEdit_1)
        mid_layout.addWidget(self.label_2)
        mid_layout.addWidget(self.label_8)
        mid_layout.addWidget(self.lineEdit_4)
        mid_layout.addStretch(1)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.label_7)
        right_layout.addWidget(self.lineEdit_2)
        right_layout.addWidget(self.pushButton0)
        right_layout.addWidget(self.pushButton1)
        right_layout.addWidget(self.pushButton3)
        right_layout.addWidget(self.pushButton2)
        right_layout.addStretch(1)

        layout = QHBoxLayout()
        layout.addLayout(left_layout)
        layout.addLayout(mid_layout)
        layout.addLayout(right_layout)
        layout.setStretchFactor(left_layout, 1)
        layout.setStretchFactor(mid_layout, 0)
        layout.setStretchFactor(right_layout, 0)

        self.lineEdit_0.setDisabled(True)
        self.lineEdit_1.setDisabled(True)
        self.lineEdit_5.setDisabled(False)
        self.lineEdit_5.setText(str(self.best_neuron_number))

        self.label_0.setText("Choose to ratio of the\ntrain-test.")
        self.label_2.setText("\n")
        self.label_3.setText("From")
        self.label_4.setText("To")
        self.label_7.setText("    How many steps\nyou want to forecast?")
        self.label_8.setText("Enter the lags:")
        self.label_9.setText("Enter the neuron number :")
        self.progress.setValue(0)

        self.setLayout(layout)

    def click_box(self, state):
        if state == Qt.Checked:
            self.optimize_state = 1  # test the best number of neuron in range
            self.lineEdit_5.setDisabled(True)
            self.lineEdit_0.setDisabled(False)
            self.lineEdit_1.setDisabled(False)
            self.label_2.setText("Best layer number is -\nIt's RMSE score is -\n")

        else:
            self.optimize_state = 0  # use the user input for number of neuron
            self.lineEdit_5.setDisabled(False)
            self.lineEdit_0.setDisabled(True)
            self.lineEdit_1.setDisabled(True)
            self.label_2.setText("\n")

    # it's done!
    def basic_predict_button(self):
        # update the ratio
        self.ratio = self.slider.value()

        if self.ratio < 20:
            self.ratio = 20
        elif self.ratio > 95:
            self.ratio = 95

        # checking the lag part
        if self.lineEdit_4.text() is not "":
            input_dim = self.lag()
            self.is_lag = 1
        else:  # no lag
            input_dim = 1
            self.is_lag = 0
            feature.data_parser(1, 1, 0)

        # checking the checkbox
        if self.optimize_state:
            self.label_2.setText("Testing the best\nnumber of neuron..")
            self.progress.setValue(5)
            self.best_neuron_number, best_rmse_score = feature.best_number_of_layer(0, input_dim)
            self.label_2.setText("\nBest neuron number is %d\nIt's RMSE score is %.2f"
                                 % (self.best_neuron_number, best_rmse_score))
        else:
            if self.lineEdit_5.text() is "":
                return -1
            else:
                try:
                    self.best_neuron_number = int(self.lineEdit_5.text())
                except ValueError:
                    return -1
                self.progress.setValue(30)
        feature.prediction(0, input_dim)

    # multiple lag?
    def n_step_forecast_button(self):
        # update the ratio
        self.ratio = self.slider.value()

        if self.ratio < 20:
            self.ratio = 20
        elif self.ratio > 95:
            self.ratio = 95

        # checking the lag part
        if self.lineEdit_4.text() is not "":
            input_dim = self.lag()
            self.is_lag = 1
        else:  # no lag
            input_dim = 1
            self.is_lag = 0
            feature.data_parser(1, 1, 0)

        # checking the n-step part
        if self.lineEdit_2.text() is "":
            return -1
        else:
            self.n_step = int(self.lineEdit_2.text())

        # checking the checkbox
        if self.optimize_state:
            self.progress.setValue(5)
            self.best_neuron_number, best_rmse_score = feature.best_number_of_layer(1, input_dim)
            self.label_2.setText("\nBest neuron number is %d\nIt's RMSE score is %.2f"
                                 % (self.best_neuron_number, best_rmse_score))
        else:
            if self.lineEdit_5.text() is "":
                return -1
            else:
                try:
                    self.best_neuron_number = int(self.lineEdit_5.text())
                except ValueError:
                    return -1
                self.progress.setValue(30)
        feature.prediction(1, input_dim)

    # all of lag?
    def recursive_forecast_button(self):
        # update the ratio
        self.ratio = self.slider.value()

        if self.ratio < 20:
            self.ratio = 20
        elif self.ratio > 95:
            self.ratio = 95

        # checking the lag part
        if self.lineEdit_4.text() is not "":
            self.is_lag = 1
            input_dim = self.lag()
        else:  # no lag
            input_dim = 1
            self.is_lag = 0
            feature.data_parser(1, 1, 0)

        # checking the n-step part
        if self.lineEdit_2.text() is "":
            return -1
        else:
            self.n_step = int(self.lineEdit_2.text())

        # checking the checkbox
        if self.optimize_state:
            self.progress.setValue(5)
            self.best_neuron_number, best_rmse_score = feature.best_number_of_layer(1, input_dim)
            self.label_2.setText("\nBest neuron number is %d\nIt's RMSE score is %.2f"
                                 % (self.best_neuron_number, best_rmse_score))
        else:
            if self.lineEdit_5.text() is "":
                return -1
            else:
                try:
                    self.best_neuron_number = int(self.lineEdit_5.text())
                except ValueError:
                    return -1
                self.progress.setValue(30)
        feature.data_parser(window.n_step, 1, 0)
        feature.prediction(3, input_dim)

    # multiple lag?
    def multi_output_forecast_button(self):
        # update the ratio
        self.ratio = self.slider.value()

        if self.ratio < 20:
            self.ratio = 20
        elif self.ratio > 95:
            self.ratio = 95

        # checking the n-step part
        if self.lineEdit_2.text() is "":
            return -1
        else:
            self.n_step = int(self.lineEdit_2.text())

        # checking the lag part
        if self.lineEdit_4.text() is not "":
            input_dim = self.lag()
            self.is_lag = 1
            input_dim = self.lags[0]
            feature.data_parser(input_dim, self.n_step, 0)
        else:  # no lag
            input_dim = 12
            self.is_lag = 0
            feature.data_parser(input_dim, self.n_step, 0)

        # checking the checkbox
        if self.optimize_state:
            self.progress.setValue(5)
            self.best_neuron_number, best_rmse_score = feature.best_number_of_layer(2, window.n_step + 1, window.n_step)
            self.label_2.setText("\nBest neuron number is %d\nIt's RMSE score is %.2f"
                                 % (self.best_neuron_number, best_rmse_score))
        else:
            if self.lineEdit_5.text() is "":
                return -1
            else:
                try:
                    self.best_neuron_number = int(self.lineEdit_5.text())
                except ValueError:
                    return -1
                self.progress.setValue(30)
        feature.prediction(2, input_dim, self.n_step)

    def lag(self):
        self.lags = str(self.lineEdit_4.text())
        self.lags = self.lags.split(',')
        try:
            for i in range(len(self.lags)):
                self.lags[i] = int(self.lags[i])

        except ValueError:
            return -1

        self.progress.setValue(30)

        if len(self.lags) is 1:
            feature.data_parser(int(self.lags[0]), 1, 0)
            return self.lags[0]
        else:
            self.lags.sort()
            feature.data_parser2(self.lags, self.ratio, len(self.lags), 1)
            return len(self.lags)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()
