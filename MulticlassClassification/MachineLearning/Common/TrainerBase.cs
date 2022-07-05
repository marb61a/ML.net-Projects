using BinaryClassification.MachineLearning.DataModels;

namespace BinaryClassification.MachineLearning.Common
{
    public class TrainerBase<TParameters> : ITrainerBase
        where TParameters : class
    {
        public string Name { get; protected set; }

        // Defines where the created model will be saved to on the file system
        protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, "classification.mdl");
        protected readonly MLContext mlContext;

        // Loads data and splits into test and train data
        protected DataOperationsCatalog.TrainTestData _dataSplit;

        // Uses the ML.Trainers package which is set in the usings file 
        protected ITrainerEstimator<MulticlassPredictionTransformer<TParameters>, TParameters> _model;
        protected ITransformer _trainedModel;

        protected TrainerBase()
        {
            mlContext = new MLContext(11);
        }

        public MultiClassificationMetrics Evaluate()
        {
            var testSetTransform = _trainedModel.Transform(_dataSplit.TestSet);

            return mlContext.MulticlassClassification.EvaluateNonCalibrated(testSetTransform);
        }

        // Check if file exists and if it does pass in the training file
        public void Fit(string trainingFileName)
        {
            if(!File.Exists(trainingFileName))
            {
                throw new FileNotFoundException($"File {trainingFileName} does not exist");
            }

            _dataSplit = LoadAndPrepareData(trainingFileName);
            var dataProcessPipeline = BuildDataProcessingPipeline();
            var trainingPipeline = dataProcessPipeline
                .Append(_model)
                .Append(MlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            _trainedModel = trainingPipeline.Fit(_dataSplit.TrainSet);
        }

        public void Save()
        {
            mlContext.Model.Save(_trainedModel, _dataSplit.TrainSet.Schema, ModelPath);
        }

        // Loads and prepares data
        private DataOperationsCatalog.TrainTestData LoadAndPrepareData(string trainingFileName)
        {
            var trainingDataView = mlContext.Data.LoadFromTextFile<PalmerPenguinsBinaryData>(trainingFileName, hasHeader: true, separatorChar: ',');
            return mlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.3);
        }

        // Build pipeline
        private EstimatorChain<NormalizingTransformer> BuildDataProcessingPipeline()
        {
             var dataProcessPipeline = MlContext.Transforms.Conversion.MapValueToKey(inputColumnName: nameof(PalmerPenguinsData.Label), outputColumnName: "Label")
                .Append(MlContext.Transforms.Text.FeaturizeText(inputColumnName: "Sex", outputColumnName: "SexFeaturized"))
                .Append(MlContext.Transforms.Text.FeaturizeText(inputColumnName: "Island", outputColumnName: "IslandFeaturized"))
                .Append(MlContext.Transforms.Concatenate("Features",
                                                "IslandFeaturized",
                                                nameof(PalmerPenguinsData.CulmenLength),
                                                nameof(PalmerPenguinsData.CulmenDepth),
                                                nameof(PalmerPenguinsData.BodyMass),
                                                nameof(PalmerPenguinsData.FliperLength),
                                                "SexFeaturized"
                                                ))
                .Append(MlContext.Transforms.NormalizeMinMax("Features", "Features"))
                .AppendCacheCheckpoint(MlContext);

            return dataProcessPipeline;
        }
    }
}