using AggressionScorerModel;
using Microsoft.ML;
using System;
using System.IO;

namespace AggressionScorer
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            Console.WriteLine("Aggression scorer model builder started.");

            var mlContext = new MLContext();
            var inputDataPreparer = mlContext
                .Transforms
                .Text
                .FeaturizeText("Features", "Comment")
                .AppendCacheCheckpoint(mlContext);

            //Create training algorithm
            Console.WriteLine("Creatong algorithm");

            var trainer = mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression();

            var trainingPipeline = inputDataPreparer.Append(trainer);

            //Load data
            Console.WriteLine("Loading data");

            var createdInputFile = @"Data\preparedInput.tsv";
            DataPreparer.CreatePreparedDataFile(createdInputFile, onlySaveSmallSubset: false);

            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                path: createdInputFile,
                hasHeader: true,
                separatorChar: '\t',
                allowQuoting: true
                );

            //Fit model
            var startTime = DateTime.Now;
            ITransformer model = trainingPipeline.Fit(trainingDataView);
            Console.WriteLine($"Model training finished in {(DateTime.Now - startTime).TotalSeconds} seconds");

            //Test model

            //Save model
            Console.WriteLine("Saving the model");

            if (!Directory.Exists("Model"))
            {
                Directory.CreateDirectory("Model");
            }
            var modelFile = @"Model\\AggressionScoreModel.zip";
            mlContext.Model.Save(model, trainingDataView.Schema, modelFile);

            Console.WriteLine($"Model is saved to {modelFile}");
        }
    }
}