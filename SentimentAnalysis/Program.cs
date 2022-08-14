using SentimentAnalysis.MachineLearning.Common;
using SentimentAnalysis.MachineLearning.DataModels;
using SentimentAnalysis.MachineLearning.Trainers;

var newSample = new SentimentData
{
    SentimentText = "This is awesome!"
};

var trainers = new List<ITrainerBase>
{
    new LbfgsLogisticRegressionTrainer(),
    new AveragedUPerceptronTrainer(),
    new PriorUTrainer(),
    new SdcaLogisticRegressionTrainer(),
    new SdcaNonCalibratedTrainer(),
    new SgdUCalibratedTrainer(),
    new SgdUNonCalibratedTrainer(),
    new DecisionTreeTrainer(5, 10),
    new DecisionTreeTrainer(5, 10, 0.1),
    new DecisionTreeTrainer(10, 20),
    new DecisionTreeTrainer(10, 20, 0.1),
    new GamTrainer(),
    new RandomForestTrainer(2, 5),
    new RandomForestTrainer(5, 10),
    new RandomForestTrainer(10, 20)
};

