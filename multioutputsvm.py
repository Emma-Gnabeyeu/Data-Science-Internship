import datetime
import io
import json
import math
import os
import time
import uuid
from abc import abstractmethod
from os.path import exists
from typing import Any

import numpy as np
import pandas as pd
import pytz
from pandas import Series, DataFrame
from pandas.core.generic import NDFrame
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import api_type
from api_type import Algorithms
from model_class import CustomModel
from parameters.env_param import env
from utils.gstorage import GStorage
from utils.manage_log import get_logger
from utils.utility_fct import psinfo, cross_product, execute_sql_query, get_table_sql

# from french_holidays import get_french_holiday_calendar

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.svm import LinearSVR
from sklearn import metrics

import pickle


class MultiOutputSVM(CustomModel):
    def __init__(self):
        """
        :param name: algorithm name
        :param target: prediction target used (quantity/margin/turnover)
        :param precision: algorithm granularity ex: store_id, all, 3600 = 1 prediction per store and hour
        :param score: score to order algorithms
        """
        super().__init__(
            name=Algorithms.multioutputsvm,
            score=0,  # high score = high priority
            target=api_type.TargetEnum.quantity,
            precision=api_type.Precision(**{
                'store': 'store_id',
                'product': 'product_id',
                'customer': 'all',
                'timestep': 3600 * 24
            })
        )
        self.n_pca = 5
        self.scaler = None
        # self.pca = None  # Not Useful
        self.components = None
        self.mean = None # mean for/of pca
        self.model = None
        self.columns = None
        self.fit_date = None
        self.last_run_date = None

    def fit(self):
        """
        fit the model (get training data, preprocess data and fit the model)
        :return: None
        """
        # if self.refit():
        #    , load the model
        # else: #the code below
        self.fit_date = pytz.timezone('Europe/Paris').localize(datetime.datetime.now()).date()

        # Load the data
        demand, product_movements = self.load_data()

        # Preprocessing the data
        x, new_products, timestep_product_store = self.preprocessing(demand, product_movements)
        self.columns = timestep_product_store.columns

        # splitting the data
        x_train, x_test, y_train, y_test = self.splitting(x, timestep_product_store, 0.0)

        # Normalise the output train set
        # self.scaler, yhat = self.standardization(y_train)
        yhat = y_train

        # PCA over the output per product per store
        my_pca = PCA(n_components=self.n_pca)
        principal_component = my_pca.fit_transform(yhat)
        columns = [f'component_{i + 1}' for i in range(self.n_pca)]
        y_train = pd.DataFrame(data=principal_component, columns=columns)
        # self.pca = my_pca
        # calculate the components of the data matrix: Transformation Matrix.
        # array [n_components, n_features] in which each row corresponds to an eigenvector;
        # the order is that of the eigenvalues.
        components = my_pca.components_
        self.components = pd.DataFrame(components, columns=self.columns,
                                       index=[f'component_{i + 1}' for i in range(self.n_pca)])
        # En fait principal_component = yhat.dot(components.T)
        mean = my_pca.mean_
        self.mean = pd.DataFrame(mean.reshape(1, -1), columns=self.columns, index=['mean'])

        # Fit the model over the Principal component
        # sv_regressor = svm.SVR(C=50, kernel='rbf', gamma=0.01)  # or LinearSVR() #
        # mul_regressor = MultiOutputRegressor(sv_regressor)
        RegModel = LinearRegression()

        # Creating the model on Training Data
        self.model = RegModel.fit(x_train, y_train)
        # # Make a test (if we consider split_size not egal to 0)
        # # PCA and Standardization Inversion (Transform data back to its original space.)
        #
        # # "predict = self.scaler.mul(self.pca.inverse_transform(prediction)) # self.scaler.T.values.dot(prediction.dot(self.components)+ self.mean)"
        #
        # # Element-wise multiplication
        # # predict = self.scaler.T.values*prediction.dot(self.components) # or np.multiply(self.scaler.T.values, prediction.dot(self.components))
        # prediction = self.model.predict(x_test)
        # # p2 = my_pca.inverse_transform(prediction)
        # predict = prediction.dot(self.components) + self.mean.values  # = my_pca.inverse_transform(prediction)
        # # Measuring Goodness of fit in Testing data 'MSE: %.3f' % MSE
        # print('R2 Value or Coefficient of determination for n_pca=%i: ' % self.n_pca)
        # print('R2_train=%f: ' % metrics.r2_score(yhat, self.model.predict(x_train).dot(self.components) + self.mean.values))
        # print('R2_test=%f: ' % metrics.r2_score(y_test, predict))
        # print('metrics training')
        # print(self.metrics_(yhat, self.model.predict(x_train).dot(self.components) + self.mean.values))
        # print('metrics test set')
        # print(self.metrics_(y_test, predict))
        # print('Worst case:')
        # print(self.metrics_(y_test, predict.clip(0, 0)))


        # Update for the news products
        # self.update_new_products_stores(new_products, timestep_product_store)
        self.update_new_products_stores_expo_smoothing(new_products, timestep_product_store)

        # Predict the components of products to come and in projects if they exist
        # That is just update the Matrix self.components and self.columns for the task of prediction
        all_store_prod = self._get_precision_col(psinfo.get_store_ids()['store_id'].tolist())
        tained_store_prod = self.components.T.reset_index()[['store_id', 'product_id']]
        # filter the products_stores not used during training.
        # join the two, keeping all of all_store_prod's indices
        joined = pd.merge(all_store_prod, tained_store_prod, on=['store_id', 'product_id'], how='left', indicator=True)
        tocome_project_products = joined.query("_merge=='left_only'").drop('_merge', axis=1)

        # tocome_project_products=all_store_prod[~(all_store_prod['store_id'].isin(tained_store_prod['store_id']) &
        #                                         all_store_prod['product_id'].isin(tained_store_prod['product_id']))]
        self.update_project_product(tocome_project_products)

    def update_new_products_stores(self, new_products, timestep_product_store):
        # 1st Method: Calling the Function for prediction over the news products (products in stores since less than 14 days)
        # Extract the real quantities over the last 14 days
        # Run a Multi Objective Optimisation or vector optimisation or multi criteria optimisation or
        # multi attribute optimisation or Pareto optimisation:
        # let's find the matrix X such that "predict \times X.T - real" is minimal,
        # just a Linear Regression
        # Since T (difference of times) is not identical for each product per stores, we are going to use a loop
        mse1 = 0
        k1 = 0
        # Definition of the coefficient alpha for exponential smoothing
        alpha = 0.85
        alpha_vector = alpha ** (
            new_products.drop_duplicates(subset=['product_id', 'store_id']).set_index(['product_id', 'store_id'])[
                'duration'])
        mse2 = 0
        # Exhibition of news products/stores ids
        arrays = [new_products['product_id'].values, new_products['store_id'].values]
        index = pd.MultiIndex.from_arrays(arrays).drop_duplicates().rename(names=['product_id', 'store_id'])
        # components for news products
        new_products_components = self.components[index]  # It is going to be updated with a parameter alpha
        end_datetime = (
            datetime.datetime.now(tz=pytz.timezone('Europe/Paris'))).date()  # .time() to point out the time,
        # for svm
        coefs1 = pd.DataFrame()
        # for pca
        coefs2 = pd.DataFrame()
        for t in range(1, new_products['duration'].max() + 1):
            dt = new_products[new_products['duration'] == t]
            if len(dt) > 0:
                start_datetime = (
                        datetime.datetime.now(tz=pytz.timezone('Europe/Paris')) - datetime.timedelta(days=t)).date()
                # 1st Method, with the model svm
                in_data = pd.date_range(start=start_datetime, end=end_datetime)
                in_data = pd.to_datetime(in_data).tz_localize(pytz.timezone('Europe/Paris'))
                in_data = pd.DataFrame(data=in_data, columns=['time']) - pd.Timedelta(days=1)
                new_sample_data_transform = self.generate_input(in_data)
                X = self.model.predict(new_sample_data_transform)
                X = pd.DataFrame(X, columns=[f'component_{i + 1}' for i in range(self.n_pca)],
                                 index=in_data['time'].dt.date)
                # Bad order of
                y = pd.pivot_table(dt, columns=['product_id', 'store_id'], index='time', values='product_count').fillna(
                    0)
                y2 = y.copy()
                y = y.merge(X, left_index=True, right_index=True)[y.columns]
                X = X.merge(y, left_index=True, right_index=True)[X.columns]
                ols = LinearRegression(fit_intercept=True)
                ols.fit(X, y)
                #TODO : 'add' intercept to mean
                y_pred = ols.predict(X)
                coef = pd.DataFrame(ols.coef_, index=y.columns, columns=X.columns)
                coefs1 = pd.concat([coefs1, coef])
                mse1 += ((y - y_pred) * (y - y_pred)).sum().sum()
                k1 += y.shape[0] * y.shape[1]

                # 2nd Method, with pca transformation, pca.transform() or just .dot() with component
                # over the entire product per store, but on the t times steps.
                x_observe = timestep_product_store[-(t + 2):-1].dot(
                    self.components.T)  # timestep_product_store[-(t+1):].dot(self.components.T).values
                # verify shape (t,n_pca)
                x_observe = pd.DataFrame(x_observe.values, columns=[f'component_{i + 1}' for i in range(self.n_pca)],
                                         index=in_data['time'].dt.date)
                y2 = y2.merge(x_observe, left_index=True, right_index=True)[y2.columns]
                x_observe = x_observe.merge(y2, left_index=True, right_index=True)[x_observe.columns]
                # optimisation, that it find X of shape (n_pca,t) such that x_observe*X-y is minimal
                linear_model = LinearRegression(fit_intercept=False)
                # or use OLS(Ordinary Least Squares) optimizer, can also use GLS Generalized Least Squares or WLS
                linear_model.fit(x_observe, y2)
                y_pred2 = linear_model.predict(x_observe)
                coef2 = pd.DataFrame(linear_model.coef_, index=y.columns, columns=x_observe.columns)
                coefs2 = pd.concat([coefs2, coef2])
                # Evaluate the error of the residual = y_true - predictedValues
                mse2 += ((y2 - y_pred2) * (y2 - y_pred2)).sum().sum()

                # modify the vector alpha
                # duration = dt['duration'].values[0]
                # alpha_time_duration = pd.DataFrame(np.array([alpha for i in range(y.shape[1])]**duration), index=y.columns,columns=['alpha'])
                # alpha_vector = pd.concat([alpha_vector, alpha_time_duration])

        mse1 = mse1 / k1
        mse2 = mse2 / k1

        #     linear_model = sm.OLS(y, X)
        #     # OLS stands for Ordinary Least Squares, can also use GLS Generalized Least Squares or WLS
        #     results = linear_model.fit()
        #     param[product_id, store_id] = results.params.tolist()
        #
        #     # Evaluate the error of the residual = y_true - predictedValues
        #     mse1 += mean_squared_error(y_true=y,
        #                                y_pred=results.fittedvalues)

        rmse = [math.sqrt(mse1), math.sqrt(mse2)]
        # Transform Index into MultiIndex
        coefs1.index = pd.MultiIndex.from_tuples(coefs1.index, names=["product_id", "store_id"])
        coefs2.index = pd.MultiIndex.from_tuples(coefs2.index, names=["product_id", "store_id"])

        # put new_products_components in the true order of columns as in coefs.index
        new_products_components = new_products_components[coefs1.index]
        # put alpha_vector in the true order of index as in coefs.index
        alpha_vector = alpha_vector.T[coefs1.index].T
        # return coefs1, coefs2, index, new_products_components, error, alpha_vector

        # Correction of the PCA's Matrix over the news products and update using the best Methods

        new_products_components_update1 = (1 - alpha_vector).T * new_products_components + alpha_vector.T * coefs1.T
        new_products_components_update2 = (1 - alpha_vector).T * new_products_components + alpha_vector.T * coefs2.T

        # Select the best one based on a metric: the norm of the residuals
        if rmse[0] < rmse[1]:
            best_new_products_components_update = new_products_components_update1
        else:
            best_new_products_components_update = new_products_components_update2

            # Update the Transformation Matrix self.components, Merge ?
            # to delete duplicates due to products in a particular store with 2 different start_date (aberration), let's do
            best_new_products_components_update = best_new_products_components_update.T.reset_index().drop_duplicates(
                subset=['product_id', 'store_id'],
                keep='first', inplace=False).set_index(['product_id', 'store_id']).T
            # and then update
        self.components[best_new_products_components_update.columns] = best_new_products_components_update

        # update the scaler for the new product
        # delete eventually duplicates in alpha_vector
        alpha_vector = alpha_vector.reset_index().drop_duplicates(subset=['product_id', 'store_id'],
                                                                  keep='first', inplace=False).set_index(
            ['product_id', 'store_id'])
        # update
        scal = self.scaler.copy().T
        scal[index] = (1 - alpha_vector).T * self.scaler.T[index] + \
                      alpha_vector.T * np.sqrt(1 / (self.scaler.T[index].shape[1]) * self.scaler.T[index] ** 2)
        self.scaler = scal.T

    def update_new_products_stores_expo_smoothing(self, new_products, timestep_product_store):
        # 1st Method: Calling the Function for prediction over the news products (products in stores since less than 14 days)
        # Extract the real quantities over the last 14 days
        # Run a Multi Objective Optimisation or vector optimisation or multi criteria optimisation or
        # multi attribute optimisation or Pareto optimisation:
        # let's find the matrix X such that "predict \times X.T - real" is minimal,
        # just a Linear Regression
        # Since T (difference of times) is not identical for each product per stores, we are going to use a loop
        mse1 = 0
        mse2 = 0
        k1 = 0
        # Definition of the coefficient alpha for exponential smoothing
        alpha = 0.85
        alpha_vector = alpha ** (
            new_products.drop_duplicates(subset=['product_id', 'store_id']).set_index(['product_id', 'store_id'])[
                'duration'])
        # Exhibition of news products/stores ids
        arrays = [new_products['product_id'].values, new_products['store_id'].values]
        index = pd.MultiIndex.from_arrays(arrays).drop_duplicates().rename(names=['product_id', 'store_id'])
        # components for news products
        new_products_components = self.components[index]  # It is going to be updated with a parameter alpha
        new_products_mean = self.mean[index]
        end_datetime = (
            datetime.datetime.now(tz=pytz.timezone('Europe/Paris'))).date()  # .time() to point out the time,
        # for svm
        coefs1 = pd.DataFrame()
        intercepts1 = pd.DataFrame()
        # for pca
        coefs2 = pd.DataFrame()
        intercepts2 = pd.DataFrame()

        for t in range(1, new_products['duration'].max() + 1):
            dt = new_products[new_products['duration'] == t]
            if len(dt) > 0:
                start_datetime = (
                        datetime.datetime.now(tz=pytz.timezone('Europe/Paris')) - datetime.timedelta(days=t)).date()
                # 1st Method, with the model svm
                in_data = pd.date_range(start=start_datetime, end=end_datetime)
                in_data = pd.to_datetime(in_data).tz_localize(pytz.timezone('Europe/Paris'))
                in_data = pd.DataFrame(data=in_data, columns=['time']) - pd.Timedelta(days=1)
                new_sample_data_transform = self.generate_input(in_data)
                X = self.model.predict(new_sample_data_transform)
                X = pd.DataFrame(X, columns=[f'component_{i + 1}' for i in range(self.n_pca)],
                                 index=in_data['time'].dt.date)
                # Bad order of
                y = pd.pivot_table(dt, columns=['product_id', 'store_id'], index='time', values='product_count').fillna(
                    0)
                y2 = y.copy()
                y = y.merge(X, left_index=True, right_index=True)[y.columns]
                X = X.merge(y, left_index=True, right_index=True)[X.columns]
                ols = LinearRegression(fit_intercept=True)
                ols.fit(X, y)
                y_pred = ols.predict(X)
                coef = pd.DataFrame(ols.coef_, index=y.columns, columns=X.columns)
                coefs1 = pd.concat([coefs1, coef])
                intercept= pd.DataFrame(ols.intercept_, index=y.columns)
                intercepts1 = pd.concat([intercepts1, intercept])
                mse1 += ((y - y_pred) * (y - y_pred)).sum().sum()
                k1 += y.shape[0] * y.shape[1]

                # 2nd Method, with pca transformation, pca.transform() or just .dot() with component
                # over the entire product per store, but on the t times steps.
                x_observe = timestep_product_store[-(t + 2):-1].dot(
                    self.components.T)  # timestep_product_store[-(t+1):].dot(self.components.T).values
                # verify shape (t,n_pca)
                x_observe = pd.DataFrame(x_observe.values, columns=[f'component_{i + 1}' for i in range(self.n_pca)],
                                         index=in_data['time'].dt.date)
                y2 = y2.merge(x_observe, left_index=True, right_index=True)[y2.columns]
                x_observe = x_observe.merge(y2, left_index=True, right_index=True)[x_observe.columns]
                # optimisation, that it find X of shape (n_pca,t) such that x_observe*X-y is minimal
                linear_model = LinearRegression(fit_intercept=True)
                # or use OLS(Ordinary Least Squares) optimizer, can also use GLS Generalized Least Squares or WLS
                linear_model.fit(x_observe, y2)
                y_pred2 = linear_model.predict(x_observe)
                coef2 = pd.DataFrame(linear_model.coef_, index=y.columns, columns=x_observe.columns)
                coefs2 = pd.concat([coefs2, coef2])
                intercept = pd.DataFrame(ols.intercept_, index=y.columns)
                intercepts2 = pd.concat([intercepts2, intercept])
                # Evaluate the error of the residual = y_true - predictedValues
                mse2 += ((y2 - y_pred2) * (y2 - y_pred2)).sum().sum()

        mse1 = mse1 / k1
        mse2 = mse2 / k1

        #     linear_model = sm.OLS(y, X)
        #     # OLS stands for Ordinary Least Squares, can also use GLS Generalized Least Squares or WLS
        #     results = linear_model.fit()
        #     param[product_id, store_id] = results.params.tolist()
        #
        #     # Evaluate the error of the residual = y_true - predictedValues
        #     mse1 += mean_squared_error(y_true=y,
        #                                y_pred=results.fittedvalues)

        rmse = [math.sqrt(mse1), math.sqrt(mse2)]
        # Transform Index into MultiIndex
        coefs1.index = pd.MultiIndex.from_tuples(coefs1.index, names=["product_id", "store_id"])
        coefs2.index = pd.MultiIndex.from_tuples(coefs2.index, names=["product_id", "store_id"])

        intercepts1.index = pd.MultiIndex.from_tuples(intercepts1.index, names=["product_id", "store_id"])
        intercepts2.index = pd.MultiIndex.from_tuples(intercepts2.index, names=["product_id", "store_id"])
        # put alpha_vector in the true order of index as in coefs.index
        alpha_vector = alpha_vector.loc[coefs1.index]
        ##########
        active_components = self.components.drop(new_products_components, axis=1).T.reset_index()
        active_mean = self.mean.drop(new_products_mean, axis=1).T.reset_index()

        # Add the categories
        products_category = psinfo.get_product_ids()
        new_products_category = new_products.drop_duplicates(subset=['product_id', 'store_id']).merge(products_category,
                                                                                                      on='product_id')

        # Average over each of the n_pca components per category per store
        new_components_mean_category = \
            active_components.merge(products_category, on='product_id').groupby(['category', 'store_id'])[
                list(set(active_components.columns) - {'store_id', 'product_id'})].mean().reset_index()

        # Components for each new product per each store
        # And delete duplicate with regard to the column product_id
        new_products_components_category = new_components_mean_category.merge(new_products_category,
                                                                              on=['category', 'store_id'],
                                                                              how='right').drop(
            ['category', 'product_count', 'duration', 'time'], axis=1).set_index(
            ['product_id', 'store_id'])

        # Replace the nan (for the new categories in new products not in the categories of the actives products) by the mean
        new_products_components_category = new_products_components_category.fillna(
            new_products_components_category.mean())

        # put new_products_components_category in the true order of index as in coefs.index
        new_products_components_category = new_products_components_category.loc[coefs1.index]

        ####### for the mean
        # Average over the mean per category per store
        new_mean_category_mean = \
            active_mean.merge(products_category, on='product_id').groupby(['category', 'store_id'])[
                list(set(active_mean.columns) - {'store_id', 'product_id'})].mean().reset_index()

        # mean for each new product per each store
        # And delete duplicate with regard to the column product_id
        new_products_mean_category = new_mean_category_mean.merge(new_products_category,
                                                                              on=['category', 'store_id'],
                                                                              how='right').drop(
            ['category', 'product_count', 'duration', 'time'], axis=1).set_index(
            ['product_id', 'store_id'])

        # Replace the nan (for the new categories in new products not in the categories of the actives products) by the mean
        new_products_mean_category = new_products_mean_category.fillna(
            new_products_mean_category.mean())

        # put new_products_components_category in the true order of index as in coefs.index
        new_products_mean_category = new_products_mean_category.loc[coefs1.index]
        ####### Now update
        # Correction of the PCA's Matrix over the news products and update using the best Methods
        new_products_components_update1 = (1 - alpha_vector).T * new_products_components_category.T + alpha_vector.T * coefs1.T
        new_products_components_update2 = (1 - alpha_vector).T * new_products_components_category.T + alpha_vector.T * coefs2.T

        # Correction of the PCA's Matrix over the news products and update using the best Methods
        new_products_mean_update1 = ((1 - alpha_vector).T * new_products_mean_category.T).reset_index(drop=True) + alpha_vector.T * intercepts1.T
        new_products_mean_update2 = ((1 - alpha_vector).T * new_products_mean_category.T).reset_index(drop=True) + alpha_vector.T * intercepts2.T

        # Select the best one based on a metric: the norm of the residuals
        if rmse[0] < rmse[1]:
            best_new_products_components_update = new_products_components_update1
            best_new_products_mean_update = new_products_mean_update1
        else:
            best_new_products_components_update = new_products_components_update2
            best_new_products_mean_update = new_products_mean_update2

            # Update the Transformation Matrix self.components, Merge ?  # names = df.columns.value_counts() to count duplicates
        # names[names > 1]
        # to delete duplicates due to products in a particular store with 2 different start_date (aberration), let's do
        best_new_products_components_update = best_new_products_components_update.T.reset_index().drop_duplicates(
            subset=['product_id', 'store_id'],
            keep='first', inplace=False).set_index(['product_id', 'store_id']).T
        # same for the mean
        best_new_products_mean_update = best_new_products_mean_update.T.reset_index().drop_duplicates(
            subset=['product_id', 'store_id'],
            keep='first', inplace=False).set_index(['product_id', 'store_id']).T
        # and then update
        self.components[best_new_products_components_update.columns] = best_new_products_components_update
        self.mean[best_new_products_mean_update.columns] = best_new_products_mean_update

        # update the scaler for the new product
        # update the scaler Matrix on the category basis
        # active_scaler = self.scaler.copy().drop(new_products_components.columns).reset_index()
        # Average over each of the std per category per store

        # new_scaler_mean_category = np.sqrt((active_scaler.set_index(['store_id', 'product_id']) ** 2).reset_index()
        #                                   .merge(products_category, on='product_id').groupby(['category', 'store_id'])[
        #                                       list(set(active_scaler.columns) - {'store_id',
        #                                                                       'product_id'})].mean()).reset_index()
        # Scaler for each product to come per each store
        # And delete duplicate with regard to the column product_id
        # new_products_scaler_category = new_scaler_mean_category.merge(new_products_category,
        #                                                             on=['category', 'store_id'],
        #                                                             how='right').drop(
        #    ['category', 'product_count', 'duration', 'time'], axis=1).set_index(
        #    ['product_id', 'store_id'])
        # Replace the nan (for the new products categories not in the categories of the active products) by the mean
        # new_products_scaler_category = new_products_scaler_category.fillna(np.sqrt((new_products_scaler_category ** 2).mean()))
        ##########
        # put new_products_scaler_category in the true order of index as in index
        # new_products_scaler_category = new_products_scaler_category.loc[index]
        # update
        # delete eventually duplicates in alpha_vector
        # alpha_vector = alpha_vector.reset_index().drop_duplicates(subset=['product_id', 'store_id'],
        #                                           keep='first', inplace=False).set_index(['product_id', 'store_id'])

        # scal = self.scaler.copy().T
        # scal[index] = (1 - alpha_vector).T * self.scaler.T[index] + \
        #              alpha_vector.T * new_products_scaler_category.T
        # self.scaler = scal.T

    def standardization(self, y_train):
        # Choose between standardization and MinMAx normalization
        # scaler = StandardScaler()
        # Storing the fit object for later reference
        # scaler = scaler.fit(y_train)
        # Generating the standardized values of X
        # yhat = scaler.transform(y_train)
        # std along the rows' axis
        scaler = y_train.std(axis=0)
        # Avoid having null numbers by replacing the std by 1 for no changes
        scaler = scaler.replace(0, 1)
        yhat = y_train.mul(1 / scaler)
        return pd.DataFrame(scaler), yhat

    def splitting(self, x: pd.DataFrame, table: pd.DataFrame, test_size=0.2):
        n = x.shape[0]
        m = table.shape[0]
        x_train = x[0:math.floor((1 - test_size) * n)]
        x_test = x[math.floor((1 - test_size) * n):n + 1]
        y_train = table[0:math.floor((1 - test_size) * m)]
        y_test = table[math.floor((1 - test_size) * m):m + 1]
        return x_train, x_test, y_train, y_test

    def preprocessing(self, df1: pd.DataFrame, df2: pd.DataFrame):
        df2['start_date'] = pd.to_datetime(df2['start_date'], utc=True).dt.tz_localize(None)
        df2['end_date'] = pd.to_datetime(df2['end_date'], utc=True).dt.tz_localize(None)

        # start since the 1st january 2022
        ref = datetime.datetime(2022, 1, 1)
        df2["start_date"] = df2["start_date"].clip(lower=ref)

        # index_names = df2[df2['end_date'] < ref].index
        # df2.drop(index_names, inplace=True)
        # for i in df2.index:
        #    if df2['start_date'][i] < ref:
        #        df2['start_date'][i] = ref

        # Create intermediates Dates over the presence of products in the different stores
        df2['time'] = df2.apply(lambda x: pd.date_range(x.start_date, x.end_date), axis=1)
        df2 = df2.explode('time').reset_index(drop=True)
        df2['time'] = df2['time'].dt.date

        # Put the dates of df1 in datetime format (from timestamp, be cautious)
        df1['time'] = pd.to_datetime(df1.time).dt.date
        df = df1.merge(df2, how='right', on=['time', 'store_id', 'product_id']).fillna({'product_count': 0})
        df = df.dropna()

        # selection of news products (products in stores since less than 14 days)
        new_products = df[df['start_date'] >= datetime.datetime.now() - datetime.timedelta(days=14)]
        # Generate the columns of duration for each of these products per store
        new_products.loc[:, 'duration'] = new_products.apply(lambda x: (datetime.datetime.now() - x.start_date).days,
                                                             axis=1)  # td1.days : le nombre de jours d'une timedelta

        # Transform into sup triangular matrix
        new_products = new_products.sort_values(by=['duration'], ascending=False)  # tri par ordre décroisant

        # Delete unnecessary columns
        new_products = new_products.drop(columns=['start_date', 'end_date'])[
            new_products.duration > 0]  # la durée T dois etre moins supérieur à 0 strictement ?

        df.drop(columns=['start_date', 'end_date'])  # , axis=1)
        # df.drop(df.loc[:, 'start_date ':'end_date '].columns, axis = 1)

        # Let's create a matrix with one row per time step and one column per product per store
        timestep_product_store = pd.pivot_table(df, values='product_count', index='time',
                                                columns=['product_id', 'store_id'],
                                                aggfunc=np.sum)  # fill_value !=0 here
        # fillna() On = 'average quantity of each product over the all data' ie Matrix completion
        # tab = df[['product_id', 'store_id', 'product_count']].groupby(['product_id', 'store_id']).mean()
        tab = timestep_product_store.mean()
        timestep_product_store = timestep_product_store.fillna(tab)

        # Pre-process the input data
        input_data = self.generate_input(df)
        return input_data, new_products, timestep_product_store

    def generate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        df['time'] = pd.to_datetime(df.time, format='%Y-%m-%d')
        input_data = pd.concat([
            df['time'].dt.date,
            df['time'].dt.day,
            # df['time'].dt.month,
            df['time'].dt.weekday,
            df['time'].dt.isocalendar().week,
        ], axis=1).drop_duplicates()
        input_data.columns = ['time', 'day', 'weekday', 'weekofyear']
        input_data = input_data.sort_values(by='time')
        return input_data[['time', 'day', 'weekday', 'weekofyear']].set_index(
            'time')  # or input_data.reset_index()[['day', 'weekday', 'weekofyear']]

    def load_data(self):
        sql_query1 = f"""
            select 
                store_id,
                product_id,
                -sum(imv.product_count+coalesce(imv2.product_count,0)) as product_count,
                date(imv.created_at) as time
            from api_storebox.inventory_movements_view imv 
            left join (
                select * from api_storebox.inventory_movements_view imv2 
            where 
                imv2.bucket= 'stock' and imv2.reason='sales_order' and imv2."source"='invoiceable_modified'
               ) imv2
            using (store_id,product_id,bucket,cart_id)
            where 
                imv.bucket= 'stock' and imv.reason='sales_order' and imv."source"='cart' and imv.created_at >= '2022-01-01'
            group by 
                product_id,
                store_id,
                date(imv.created_at) 
        """
        df1 = get_table_sql(sql_query1, table_name='demand').query('product_count<=20')
        if len(df1) == 0:
            df1 = pd.DataFrame(columns=['product_id', 'store_id', 'time', 'product_count'])
        df1['product_count'] = df1.product_count.clip(lower=0)
        sql_query2 = f"""
            select 
                store_id,
                product_id,
                start_date,
                end_date
            from 
                api_storebox.product_movements 
            where 
                end_date>= date('2022-01-01') 
        """
        df2 = get_table_sql(sql_query2, table_name='product_movements')
        if len(df2) == 0:
            df2 = pd.DataFrame(columns=['store_id', 'product_id', 'start_date', 'end_date'])
        return df1, df2

    def predict(self, X):
        """
        predit data using current model
        :param X:pd.DataFrame: 3 cols : 'time', 'store_id', 'product_id' - one row per (timestep x store_id x product_id)   ## time==date
        :return: pd.DataFrame: 4 cols : 'time', 'store_id', 'product_id', self.target
        """

        # X['time'] = pd.to_datetime(X['time'], utc=True).dt.tz_convert('Europe/Paris')
        # store_ids = list(X['store_id'].unique())
        # InputData being a sequence of date
        input_data = X
        data_transform = self.generate_input(input_data)
        # Generating Predictions
        predict = self.model.predict(data_transform)
        # PCA and Standardization Inversion (Transform data back to its original space.)

        # En fait self.pca.inverse_transform(prediction)=predict.dot(self.components.T)^_1 : np. linalg. inv(components.T) =components car Matrice Orthogonale
        # prediction1 = self.scaler.inverse_transform(self.pca.inverse_transform(predict)) is essentially done by the line code below
        # prediction = self.scaler.inverse_transform(predict.dot(self.components))

        # decomend to predict over all the product_stores
        # prediction = np.multiply(self.scaler.T.values, predict.dot(self.components))
        # Let's consider only those from X we are interested in.
        sp = X[['product_id', 'store_id']].drop_duplicates()
        # prediction = np.multiply(self.scaler.loc[pd.MultiIndex.from_frame(sp)].T.values,
        #                         predict.dot(self.components.T.loc[pd.MultiIndex.from_frame(sp)].T))
        prediction = predict.dot(self.components.T.loc[pd.MultiIndex.from_frame(sp)].T) + self.mean[pd.MultiIndex.from_frame(sp)].values  ### to see
        #TODO add pca.mean_

        # Transform to DataFrame
        # decomend to predict over all the product_stores
        # prediction = pd.DataFrame(data=prediction, columns=self.components.columns, index=data_transform.index) #columns=multiindex or columns=self.columns
        # for just the X product_stores,
        prediction = pd.DataFrame(data=prediction,
                                  columns=self.components.T.loc[pd.MultiIndex.from_frame(sp)].T.columns,
                                  index=data_transform.index)

        # inverse operation of pivot_table()
        prediction = prediction.stack([0, 1]).reset_index().rename({0: self.target.value}, axis=1)

        # apply relu function to prediction = product_count
        prediction[self.target.value] = prediction[self.target.value].clip(lower=0)
        prediction['time'] = prediction.time.astype('datetime64[ns, Europe/Paris]')
        prediction_result = prediction.drop_duplicates(['time', 'store_id', 'product_id']).merge(X,
                                                                                                 on=['time', 'store_id',
                                                                                                     'product_id'],
                                                                                                 how='right')
        return prediction_result[['time', 'store_id', 'product_id', self.target.value]]

    def update_project_product(self, tocome_project_products):
        """
           Mean over categories
        """
        # Exhibition of projects news products/stores ids:

        # project_products = get_table_sql("""
        # select distinct sv.store_id,mpm.product_id
        # from google_sheets.merch_product_movements mpm
        # join api_storebox.stores_view sv
        # using (cluster)
        # where start_date >= now()-interval '1week'
        # """, table_name='to_come')

        new_components = self.components.copy().T.reset_index()
        new_mean = self.mean.copy().T.reset_index()

        # Add the categories
        products_category = psinfo.get_product_ids()
        project_products_category = tocome_project_products.merge(products_category, on='product_id')

        # Average over each of the n_pca components per category per store
        new_components_mean_category = \
            new_components.merge(products_category, on='product_id').groupby(['category', 'store_id'])[
                list(set(new_components.columns) - {'store_id', 'product_id'})].mean().reset_index()

        # Average over the means per category per store
        new_mean_category_mean = \
            new_mean.merge(products_category, on='product_id').groupby(['category', 'store_id'])[
                list(set(new_mean.columns) - {'store_id', 'product_id'})].mean().reset_index()

        # Components for each product to come per each store
        # And delete duplicate with regard to the column product_id
        new_project_components_category = new_components_mean_category.merge(project_products_category,
                                                                             on=['category', 'store_id'],
                                                                             how='right').drop(
            ['category'], axis=1).set_index(
            ['product_id', 'store_id'])
        # Replace the nan (for the new categories not in the categories of the training set) by the mean
        new_project_components_category = new_project_components_category.fillna(new_project_components_category.mean())

        ## same for the means
        new_project_mean_category = new_mean_category_mean.merge(project_products_category,
                                                                             on=['category', 'store_id'],
                                                                             how='right').drop(
            ['category'], axis=1).set_index(
            ['product_id', 'store_id'])
        # Replace the nan (for the new categories not in the categories of the training set) by the mean
        new_project_mean_category = new_project_mean_category.fillna(new_project_mean_category.mean())

        # Concatenate over the columns
        tmp = self.components.T
        tmp = tmp.drop(new_project_components_category.index, errors='ignore')
        # tmp = new_project_components_category.drop(tmp.index, errors='ignore') ?
        tmp = pd.concat([tmp, new_project_components_category])
        self.components = tmp.T
        # same for the means
        tmp2 = self.mean.T
        tmp2 = tmp2.drop(new_project_mean_category.index, errors='ignore')
        tmp2 = pd.concat([tmp2, new_project_mean_category])
        self.mean = tmp2.T

        # update the columns
        self.columns = pd.MultiIndex.from_frame(pd.concat([pd.DataFrame(self.columns),
                                                           pd.DataFrame(new_project_components_category.index)
                                                           ]).drop_duplicates())

        # update the scaler Matrix on the category basis
        # new_scaler = self.scaler.copy().reset_index()
        # Average over each of the std per category per store

        # new_scaler_mean_category = np.sqrt((new_scaler.set_index(['store_id', 'product_id'])**2).reset_index()
        #                                   .merge(products_category, on='product_id').groupby(['category', 'store_id'])[
        #        list(set(new_scaler.columns) - {'store_id', 'product_id'})].mean()).reset_index()
        # Scaler for each product to come per each store
        # And delete duplicate with regard to the column product_id
        # new_project_scaler_category = new_scaler_mean_category.merge(project_products_category,
        #                                                             on=['category', 'store_id'],
        #                                                             how='right').drop(
        #    ['category'], axis=1).set_index(
        #    ['product_id', 'store_id'])
        # Replace the nan (for the new categories not in the categories of the training set) by the mean
        # new_project_scaler_category = new_project_scaler_category.fillna(np.sqrt((new_project_scaler_category**2).mean()))
        # Concatenate over the columns
        # tmp2 = self.scaler
        # tmp2 = new_project_scaler_category.drop(tmp2.index, errors='ignore')

        # self.scaler = pd.concat([self.scaler, tmp2])

    def save(self):
        """
        :return: dict of file_name:file of type str:io.BytesIO to be saved
        """
        files = {}
        final_svm_model = self.model
        file_write_stream = io.BytesIO()
        pickle.dump(final_svm_model, file_write_stream)
        file_write_stream.seek(0)
        files['model'] = file_write_stream

        final_scaler = self.scaler
        file_write_stream = io.BytesIO()
        pickle.dump(final_scaler, file_write_stream)
        file_write_stream.seek(0)
        files['scaler'] = file_write_stream

        final_n_pca = self.n_pca
        file_write_stream = io.BytesIO()
        pickle.dump(final_n_pca, file_write_stream)
        file_write_stream.seek(0)
        files['n_pca'] = file_write_stream

        final_columns = self.columns
        file_write_stream = io.BytesIO()
        pickle.dump(final_columns, file_write_stream)
        file_write_stream.seek(0)
        files['columns'] = file_write_stream

        final_components = self.components
        file_write_stream = io.BytesIO()
        pickle.dump(final_components, file_write_stream)
        file_write_stream.seek(0)
        files['components'] = file_write_stream

        final_mean = self.mean
        file_write_stream = io.BytesIO()
        pickle.dump(final_mean, file_write_stream)
        file_write_stream.seek(0)
        files['mean'] = file_write_stream

        fit_date = self.fit_date
        file_write_stream = io.BytesIO()
        pickle.dump(fit_date, file_write_stream)
        file_write_stream.seek(0)
        files['fit_date'] = file_write_stream

        return files

    def load(self, files):
        """
        :param files: dict of file_name:files
        :return:
        """
        """if self.refit() != False:
            self.fit()
        else:"""
        self.model = pickle.load(files['model'])
        self.scaler = pickle.load(files['scaler'])
        self.n_pca = pickle.load(files['n_pca'])
        self.columns = pickle.load(files['columns'])
        self.components = pickle.load(files['components'])
        self.mean = pickle.load(files['mean'])
        self.fit_date = pickle.load(files['fit_date'])

    def refit(self) -> bool:
        """
        should the algorithms be refitted ?
        :return: bool
        """
        if self.fit_date == pytz.timezone('Europe/Paris').localize(datetime.datetime.now()).date():
            return False
        else:
            return True

    def validate_new_model(self) -> bool:
        """
        is the new algorithm better than previous one ?
        :return: bool
        """
        # supposons qu'il ya toujours entrainement avant de tester la validation
        # Mettre un attribut loss,
        # ré-évaluer automatiquement ses prévisions après les avoir comparer à la réalité pour intégrer les écarts éventuellement constaté
        # comparer les deux loss et valider le model de plus faible loss,
        return True

    # For test in case we set text_size not equal to zero
    def metrics_(self, orig, prediction):
        mse = mean_squared_error(y_true=orig,
                                 y_pred=prediction)
        mae = mean_absolute_error(y_true=orig,
                                  y_pred=prediction)
        ms_le = mean_squared_log_error(y_true=np.abs(orig),
                                       y_pred=np.abs(prediction))
        metric = dict()
        metric['MSE'] = mse
        metric['MAE'] = mae
        metric['MSLE'] = ms_le
        return metric


if __name__ == '__main__':
    model = MultiOutputSVM()
    model.fit()
    # Calling the functions for some applications
    start_date = datetime.datetime(2022, 9, 6)
    end_date = datetime.datetime(2022, 9, 9)
    data = pd.date_range(start=start_date, end=end_date)
    data = pd.to_datetime(data).tz_localize(pytz.timezone('Europe/Paris'))
    NewSampleData = pd.DataFrame(data=data,
                                 columns=['time'])
    NewSampleData['product_id'] = '5d0794866885630014681ba6' # '5f8ee80289bc4b000417840f'  #
    NewSampleData['store_id'] = '5f1198de8746ae00042869ba'  # '5f1198de8746ae00042869ba'   #

    # Calling the Function for prediction
    d1 = model.predict(NewSampleData)
    print(d1)
    # tester le load() et le save()
    model2 = MultiOutputSVM()
    model2.load(model.save())
    d2 = model2.predict(NewSampleData)
    assert (d1.equals(d2))
    print(d1.equals(d2))

    model3 = MultiOutputSVM()
    model.save_wrapper()
    model3.load_wrapper()
    d3 = model2.predict(NewSampleData)
    assert (d1.equals(d3))
    print(d1.equals(d3))
