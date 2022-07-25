using SVM.MachineLearning.Common;

namespace SVM.MachineLearning.Trainers
{
    public class LinearSvmTrainer: TrainerBase<LinearBinaryModelParameters>
    {
        public LinearSvmTrainer(): base()
        {
            Name= "Linear SVM";
            _model = mlContext.BinaryClassification.Trainers.LinearSvm(labelColumnName: "Label");
        }
    }
}