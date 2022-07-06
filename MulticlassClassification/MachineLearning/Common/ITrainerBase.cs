namespace MulticlassClassification.MachineLearning.Common;

public interface ITrainerBase
{
    string Name { get; }
    void Fit(string trainingFileName);

    MulticlassClassificationMetrics Evaluate();
    void Save();
}