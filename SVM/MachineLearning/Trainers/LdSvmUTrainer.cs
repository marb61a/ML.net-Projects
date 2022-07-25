using SVM.MachineLearning.Common;

namespace SVM.MachineLearning.Trainers
{
    public class LdSvmUTrainer: TrainerBase<LdSvmModelParameters>
    {
        public LdSvmUTrainer(int treeDepth): base()
        {
            Name = $"LD-SVM with {treeDepth} tree depth";
            _model = mlContext.BinaryClassification.Trainers.LdSvm(labelColumnName: "Label", treeDepth: treeDepth);
        }
    }
}