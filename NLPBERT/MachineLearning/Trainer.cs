using NLPBERT.MachineLearning.DataModel;

namespace NLPBERT.MachineLearning
{
    public class Trainer
    {
        private readonly MLContext _mlContext;

        public Trainer()
        {
            _mlContext = new MLContext(11);
        }

        // Model Pipeline
        public ITransformer BuildAndTrain(string bertModelPath)
        {
            var pipeline = _mlContext.Transforms.ApplyOnnxModel(
                modelFile: bertModelPath,
                outputColumnNames: new[]
                {
                    "unstack:1",
                    "unstack:0",
                    "unique_ids:0"
                },
                inputColumnNames: new[]
                {
                    "unique_ids_raw_output___9:0",
                    "segment_ids:0",
                    "input_mask:0",
                    "input_ids:0"
                },
                gpuDeviceId: 0
            );

            return pipeline.Fit(_mlContext.Data.LoadFromEnumerable(new List<BertInput>()));
        }
    }
}