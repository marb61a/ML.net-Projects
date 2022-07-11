using DecisionTreeClassification.MachineLearning.DataModels;

namespace DecisionTreeClassification.MachineLearning.Common;

// This is the base class for trainers
public abstract class TrainerBase<TParameters>: ITrainerBase where TParameters: class
{
    
}