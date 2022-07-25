using SVM.MachineLearning.Common;

namespace SVM.MachineLearning.Trainers
{
    public class LinearSvmUTrainer: TrainerBase<LinearBinaryModelParameters>
    {
        public LinearSvmUTrainer(): base()
        {
            Name= "Linear SVM";
            _model = mlContext.BinaryClassification.Trainers.LinearSvm(labelColumnName: "Label");
        }
    }
}