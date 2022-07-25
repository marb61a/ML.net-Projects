using SVM.MachineLearning.Common;

namespace SVM.MachineLearning.Trainers
{
    public class LdSvmTrainer: TrainerBase<LdSvmModelParameters>
    {
        public LdSvmTrainer(int treeDepth): base()
        {
            Name = $"LD-SVM with {treeDepth} tree depth";
            _model = mlContext.BinaryClassification.Trainers.LdSvm(labelColumnName: "Label", treeDepth: treeDepth);
        }
    }
}