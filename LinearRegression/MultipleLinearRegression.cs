using System;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace LinearRegressionBasic
{
    public class MultipleLinearRegression
    {
        public double _b;
        public double[] _w;

        public MultipleLinearRegression()
        {
            _b = 0;

        } 

        // For training
        public void Fit(double[,] X, double[,] y)
        {
            var input = ExtendInputWithOnes(X);

            // Converts y to matrix doubles
            var output= Matrix<double>.Build.DenseOfArray(y);

            var coeficients = ((input.Transpose() * input).Inverse() * input.Transpose() * output).Transpose().Row(0);

            _b = coeficients.ElementAt(0);
            _w = SubArray(coeficients.ToArray(), 1, X.GetLength(1));
        }

        public double Predict(double[,] x)
        {
            var input = Matrix<double>.Build.DenseOfArray(x).Transpose();
            var w = Vector<double>.Build.DenseOfArray(_w);
            return input.Multiply(w).ToArray().Sum() + _b;

        }

        private Matrix<double> ExtendInputWithOnes(double[,] X)
        {
            // This adds ones to the input array which models coefficient b in the data
            var ones = Matrix<double>.Build.Dense(X.GetLength(0), 1, 1d);
            var extendedX = ones.Append(Matrix<double>.Build.DenseOfArray(X));

            return extendedX;
        }

        private double[] SubArray(double[] data, int index, int length)
        {
            double[] result = new double[length];
            Array.Copy(data, index, result, 0, length);

            return result;
        }
    }
}