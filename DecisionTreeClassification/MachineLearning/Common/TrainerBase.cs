using DecisionTreeClassification.MachineLearning.DataModels;

namespace DecisionTreeClassification.MachineLearning.Common;

// This is the base class for trainers
public abstract class TrainerBase<TParameters>: ITrainerBase where TParameters: class
{
     public string Name { get; protected set; }

    protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, "classification.mdl");

    protected readonly MLContext mlContext;

    protected DataOperationsCatalog.TrainTestData _dataSplit;
    protected ITrainerEstimator<BinaryPredictionTransformer<TParameters>, TParameters> _model;
    protected ITransformer _trainedModel;

    protected TrainerBase()
    {
        mlContext = new MLContext(11);
    }

    public void Fit(string trainingFileName)
    {
        if(!File.Exists(trainingFileName))
        {
            throw new FileNotFoundException($"File {trainingFileName} doesn't exist.");
        }

        _dataSplit = LoadAndPrepareData(trainingFileName);
        var dataProcessPipeline = BuildDataProcessingPipeline();
        var trainingPipeline = dataProcessPipeline.Append(_model);
        _trainedModel = trainingPipeline.Fit(_dataSplit.TrainSet);
    }

    public BinaryClassificationMetrics Evaluate()
    {
        var testSetTransform = _trainedModel.Transform(_dataSplit.TestSet);
        return mlContext.BinaryClassification.EvaluateNonCalibrated(testSetTransform);
    }

    public void Save()
    {
        mlContext.Model.Save(_trainedModel, _dataSplit.TrainSet.Schema, ModelPath);
    }


    private EstimatorChain<NormalizingTransformer>BuildDataProcessingPipeline()
    {
        var dataProcessPipeline = mlContext.Transforms.Concatenate("Features",
                nameof(PalmerPenguinsBinaryData.CulmenDepth),
                nameof(PalmerPenguinsBinaryData.CulmenLength)
            )
            .Append(mlContext.Transforms.NormalizeMinMax("Features", "Features"))
            .AppendCacheCheckpoint(mlContext);

        return dataProcessPipeline;
    }

    private DataOperationsCatalog.TrainTestData LoadAndPrepareData(string trainingFileName)
    {
        var trainingDataView = mlContext.Data.LoadFromTextFile<PalmerPenguinsBinaryData>(trainingFileName, hasHeader: true, separatorChar: ',');
        return mlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.3);
    }

}