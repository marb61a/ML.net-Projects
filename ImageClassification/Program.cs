using ImageClassification.MachineLearning;

// Add path to data file to be loaded
var dataLoader = new DataLoader("");

var trainer = new ImageClassifier(ImageClassificationTrainer.Architecture.MobilenetV2, dataLoader);
trainer.Fit();
trainer.Save();

var predictor = new Predictor(dataLoader);

Console.WriteLine("Make predictions on test dataset:");
var predictions = predictor.MakeTestDatasetPredictions();

foreach (var prediction in predictions)
{
    string imageName = Path.GetFileName(prediction.ImagePath);
    Console.WriteLine($"Image: {imageName} | Actual Value: {prediction.Label} | Predicted Value: {prediction.PredictedLabel}");
}