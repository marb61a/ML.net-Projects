namespace BinaryClassification.MachineLearning.DataModels;

// Models Palmer Penguins Binary Data
public class PalmerPenguinsBinaryData
{
    [LoadColumn(0)]
    public bool Label { get; set; }

    [LoadColumn(2)]
    public bool CulmenLength { get; set; }

    [LoadColumn(3)]
    public bool Culmendepth { get; set; }
}