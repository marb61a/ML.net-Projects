using LinearRegressionBasic;
using System;
using System.Linq;

float[] X = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
float[] y = { 6, 6, 11, 17, 16, 20, 23, 23, 29, 33, 39 };

var linearRegression = new LinearRegression();
linearRegression.Fit(X, y);

var prediction = linearRegression.Predict(X);

Console.WriteLine("Simple Linear Regression Predictions:");
Console.WriteLine($"{string.Join(", ", prediction.Select(p => p.ToString()))}");

Console.WriteLine("Actual Value:");
Console.WriteLine($"{string.Join(", ", y.Select(p => p.ToString()))}");

double[,] mX = { { 1, 2, 3},
                { 2, 9, 11},
                { 56, 111, 66}};

double[,] my = { { 6 }, { 6 }, { 11 } };

var multipleLinearRegression = new MultipleLinearRegression();
multipleLinearRegression.Fit(mX, my);
var predictions = multipleLinearRegression.Predict(new double[,] { { 3 }, { 5 }, { 7 } });

Console.WriteLine($"Multiple Linear Regression Prediction: {predictions}");

Console.ReadLine();
