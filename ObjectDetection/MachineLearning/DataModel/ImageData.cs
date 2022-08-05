namespace ObjectDetection.MachineLearning.DataModel
{
    public class ImageData
    {
        [ColumnName("image")]
        [ImageType(416, 416)]
        public Bitmap Image { get; set; }

        [ColumnName("width")]
        public float ImageWidth => Image.Width;

        [ColumnName("height")]
        public float ImageHeight => Image.Height;
    }
}