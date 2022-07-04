namespace BinaryClassification.MachineLearning.DataModels;

// Models Palmer Penguins Binary Data
public class PalmerPenguinsBinaryData
{
    [LoadColumn(0)]
    public bool Label { get; set; }

    [LoadColumn(2)]
    public float CulmenLength { get; set; }

    [LoadColumn(3)]
    public float CulmenDepth { get; set; }
}