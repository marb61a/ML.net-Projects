using RandomForestClassification.MachineLearning.DataModels;

namespace RandomForestClassification.MachineLearning.Common;

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

}