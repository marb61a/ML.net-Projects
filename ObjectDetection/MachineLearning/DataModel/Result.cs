namespace ObjectDetection.MachineLearning.DataModel
{
    public class Result{
        // Get x1, y1, x2, y2 page coordinates
        public float[] BoundingBox { get; }

        // Bounding box category
        public string Label { get;}

        // Confidence level
        public float Confidence { get; }

        public Result(float[] boundingBox, string label, float confidence)
        {
            BoundingBox = boundingBox;
            Label = label;
            Confidence = confidence;
        }
    }
}