namespace ObjectDetection.MachineLearning.DataModel
{
    public class ImagePrediction
    {
        // Yolo config available at the following url
        // https://github.com/hunglc007/tensorflow-yolov4-tflite/blob/9f16748aa3f45ff240608da4bd9b1216a29127f5/core/config.py#L18

        // Defines size of anchor boxes
        private readonly float [][][] Anchors = new float [][][]
        {
            new float[][] { new float[] { 12, 16 }, new float[] { 19, 36 }, new float[] { 40, 28 } },
            new float[][] { new float[] { 36, 75 }, new float[] { 76, 55 }, new float[] { 72, 146 } },
            new float[][] { new float[] { 142, 110 }, new float[] { 192, 243 }, new float[] { 459, 401 } }
        };

        private readonly float[] STRIDES = new float[] { 8, 16, 32 };
        private readonly float[] XYSCALE = new float[] { 1.2f, 1.1f, 1.05f };
        private readonly int[] SHAPES = new int[] { 52, 26, 13 };


        private const int _anchorsCount = 3;
        private const float _scoreThreshold = 0.5f;
        private const float _iouThreshold = 0.5f;

        // Output results
        [VectorType(1, 52, 52, 3, 85)]
        [ColumnName("Identity:0")]
        public float [] Identity { get; set; }

        [VectorType(1, 26, 26, 3, 85)]
        [ColumnName("Identity_1:0")]
        public float [] Identity1 { get; set; }

        [VectorType(1, 13, 13, 3, 85)]
        [ColumnName("Identity_2:0")]
        public float [] Identity2 { get; set; }

        [ColumnName("width")]
        public float ImageWidth { get; set; }

        [ColumnName("height")]
        public float ImageHeight { get; set; }

        public IReadOnlyList<Result> GetResults(string[] categories)
        {
            // Applies non-max suppression and returns a list of results
        }

        // Covers postprocess_bbbox
        // https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov4#postprocessing-steps
        private List<float[]> PostProcessBoundingBoxes(float[][] results, int classesCount)
        {
            List<float> postProcessedResults = new List<float>();
            
        }
    }
}