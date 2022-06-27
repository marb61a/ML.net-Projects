namespace LinearRegression
{
    public class LinearRegression
    {
        public float _b0;
        public float _b1;

        public LinearRegression()
        {
            _b0 = 0;
            _b1 = 0;
        }

        public void Fit(float[] X, float [] y)
        {
            var ssxy = X.Zip(y, (a, b) => a * b).Sum() - X.Length * X.Average() * y.Average();
            var ssxx = X.Zip(X, (a, b) => a * b).Sum() - X.Length * X.Average() * X.Average();

            _b1 = ssxy/ssxx;
            _b0 = y.Average() - _b1 * X.Average();
        }

        public float[] Predict(float[] z)
        {
            return z.Select(i => _b0 + i * _b1).ToArray();
        }
    }
}