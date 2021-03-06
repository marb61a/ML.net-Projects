using MulticlassClassification.MachineLearning.Common;

namespace MulticlassClassification.MachineLearning.Trainers;

public class NaiveUBayesTrainer : TrainerBase<NaiveBayesMulticlassModelParameters>
{
    public NaiveUBayesTrainer() : base()
    {
        Name = "Naive Bayes";
        _model = mlContext.MulticlassClassification.Trainers
                    .NaiveBayes(labelColumnName: "Label", featureColumnName: "Features");
    }
}