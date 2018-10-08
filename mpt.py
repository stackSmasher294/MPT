# import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Stock(object):

    def __init__(self, name, csv_file):
        self.name = name
        fileobj = open(csv_file)
        self.csv_reader = csv.DictReader(fileobj)
        self.returns_list = []
        closing_prices = [ float(row['Adj Close']) for row in self.csv_reader]
        self.closing_prices = closing_prices
        self.returns_list = [ ((closing_prices[i + 1]/closing_prices[i]) - 1) * 100 for i in range(len(closing_prices) - 1)]
 
    def show_closing_prices(self):
        plt.grid(True)
        plt.plot(self.closing_prices)
        plt.show()

    def show_distribution(self):
        # hist, edges = np.histogram(self.returns_list, bins=100)
        plt.grid(True)
        # plt.plot(hist)
        # plt.show()
        # plt.figure()
        plt.subplot(211)
        plt.grid(True)
        plt.title(self.name + ' : Histogram of returns (in %)')
        plt.xlabel('Daily Return (%)')
        plt.ylabel('Frequency')
        plt.hist(self.returns_list, bins=100)


        plt.subplot(212)
        plt.grid(True)
        plt.title(self.name + ' :Gaussian PDF of returns (in %)')
        plt.xlabel('Daily Return (%)')
        plt.ylabel('Probablity Density')
        sns.kdeplot(self.returns_list)
        plt.show()

    def get_mean_variance(self):
        N = len(self.returns_list)
        variance = np.var(self.returns_list)
        return (sum(self.returns_list) / N, variance)
    
    def get_net_return(self):
        opening = self.closing_prices[0]
        close = self.closing_prices[-1]
        return ((close/opening) - 1) * 100

class Porfolio(object):

    def __init__(self, stock_list_obj):
        self.stock_list = stock_list_obj
        self.n_stocks = len(stock_list_obj)
        # assign equal weights to all the stocks to start with .. 
        self.weight_list = [1./self.n_stocks for i in range(self.n_stocks)]
        self.mean_vector = np.array([stock.get_mean_variance()[0] for stock in self.stock_list])
        self.return_matrix = np.array([ stock.returns_list for stock in self.stock_list ])
        print np.shape(self.return_matrix)
        self.returns_covariance = np.cov(self.return_matrix)
        print self.returns_covariance

    def set_portfolio_weights(self, weight_list):
        self.weight_list = weight_list

    def get_portfolio_mean(self):
        self.weight_vector = np.array([self.weight_list])
        expected_return = np.matmul(self.mean_vector, self.weight_vector.T)
        return expected_return[0] # only one element

    def get_portfolio_variance(self):
        N, N = np.shape(self.returns_covariance) 
        # Calculate a mask which has to be dot multiplied with the covariance matrix
        mask = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                mask[i][j] = self.weight_list[i] * self.weight_list[j]
        # dot multiply mask with covariance
        weighted_covariance = mask * self.returns_covariance
        return np.sum(weighted_covariance)

    def get_portfolio_performance(self, weight_list):
        self.set_portfolio_weights(weight_list)
        mean = self.get_portfolio_mean()
        var = self.get_portfolio_variance()
        return mean, var


if __name__ == '__main__':
    root_dir = 'daily'
    csv_list = os.listdir(root_dir)

    stocklist = []
    for csv_file in csv_list:
        filename = root_dir + '/' + csv_file
        stocklist.append(Stock(csv_file.split('.')[0], filename))

    for stock in stocklist:
        print stock.name
        print '\tMean, Variance: {}'.format(stock.get_mean_variance())
        print '\tNet return: {}'.format(stock.get_net_return())
        print '\tpoints: {}'.format(len(stock.returns_list))
        # stock.show_closing_prices()
        # stock.show_distribution()
    portfolio = Porfolio(stocklist)

    print portfolio.get_portfolio_performance([0.25, 0.25, 0.25, 0.25])
    print portfolio.get_portfolio_performance([0.1, 0.2, 0.3, 0.4])
    