﻿using AggressionScorerModel;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers;
using System;
using System.IO;
using System.Linq;

namespace AggressionScorer
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            //CrossValidate();
            //return;

            Console.WriteLine("Aggression scorer model builder started.");

            var mlContext = new MLContext(0);

            //Load data
            Console.WriteLine("Loading data");

            var createdInputFile = @"Data\preparedInput.tsv";
            DataPreparer.CreatePreparedDataFile(createdInputFile, onlySaveSmallSubset: true);

            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                path: createdInputFile,
                hasHeader: true,
                separatorChar: '\t',
                allowQuoting: true
                );

            var inputDataSplit = mlContext.Data.TrainTestSplit(trainingDataView, testFraction: .2, seed: 0);

            //Build pipeline
            var inputDataPreparer = mlContext
                .Transforms
                .Text
                .FeaturizeText("Features", "Comment")
                .Append(mlContext.Transforms.NormalizeMeanVariance("Features"))
                .AppendCacheCheckpoint(mlContext)
                .Fit(inputDataSplit.TrainSet);

            //Create training algorithm
            Console.WriteLine("Creating algorithm");

            var trainer = mlContext
                .BinaryClassification
                .Trainers
                .LbfgsLogisticRegression();

            //Fit model
            Console.WriteLine("Training model");

            var startTime = DateTime.Now;
            var transformedData = inputDataPreparer.Transform(inputDataSplit.TrainSet);
            ITransformer model = trainer.Fit(transformedData);

            Console.WriteLine($"Model training finished in {(DateTime.Now - startTime).TotalSeconds} seconds");

            //Test model
            EvaluateModel(mlContext, model, inputDataPreparer.Transform(inputDataSplit.TestSet));

            //Save model
            Console.WriteLine("Saving the model");

            if (!Directory.Exists("Model"))
            {
                Directory.CreateDirectory("Model");
            }
            var modelFile = @"Model\\AggressionScoreModel.zip";
            mlContext.Model.Save(model, trainingDataView.Schema, modelFile);

            Console.WriteLine($"Model is saved to {modelFile}");

            var dataPreparePipelineFile = "dataPreparePipeline.zip";

            Console.WriteLine("Saving the input data preparing pipeline");

            mlContext.Model.Save(inputDataPreparer, trainingDataView.Schema, dataPreparePipelineFile);

            Console.WriteLine($"The pipeline is saved to {dataPreparePipelineFile}");

            var retrainedModel = RetrainModel(modelFile, dataPreparePipelineFile);

            var completeRetrainedPipeline = inputDataPreparer.Append(retrainedModel);

            Console.WriteLine($"Saving the retrained model");
            string retrainedModelFile = @"Model\\AggressionScoreRetrainedModel.zip";

            mlContext.Model.Save(completeRetrainedPipeline, trainingDataView.Schema, retrainedModelFile);

            Console.WriteLine("The model is saved to {0}", retrainedModelFile);

            EvaluateModel(mlContext, completeRetrainedPipeline, inputDataSplit.TestSet);
        }

        private static void CrossValidate()
        {
            var mlContext = new MLContext(0);

            //Creating a small data set for faster cross validation
            string createdInputFile = @"Data\preparedInput.tsv";
            DataPreparer.CreatePreparedDataFile(createdInputFile, onlySaveSmallSubset: true);

            //Load the data from the file into a DataView
            IDataView inputDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                path: createdInputFile,
                hasHeader: true,
                separatorChar: '\t',
                allowQuoting: true
            );

            // Prepare input data to a form consumable by a machine learning model
            var dataPipeline = mlContext
                .Transforms
                .Text
                .FeaturizeText("Features", "Comment")
                .Append(mlContext.Transforms.NormalizeMeanVariance("Features"))
                .AppendCacheCheckpoint(mlContext);

            // Create the training algorithms
            var trainers = new IEstimator<ITransformer>[]
            {
                mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(),
                mlContext.BinaryClassification.Trainers.SgdCalibrated()
            }.Reverse();

            foreach (var trainer in trainers)
            {
                var modelPipeline = dataPipeline.Append(trainer);

                var crossValidationResults = mlContext
                    .BinaryClassification
                    .CrossValidate(inputDataView, modelPipeline, numberOfFolds: 5);

                var averageAccuracy = crossValidationResults.Average(m => m.Metrics.Accuracy);
                Console.WriteLine($"Cross validated average accuracy: {averageAccuracy:0.###}");

                var averageF1Score = crossValidationResults.Average(m => m.Metrics.F1Score);
                Console.WriteLine($"Cross validated average F1Score: {averageF1Score:0.###}");
            }
        }

        private static ITransformer RetrainModel(string modelFile, string dataPreparePipelineFile)
        {
            MLContext mlContext = new MLContext(0);

            // Load pre trained model
            ITransformer pretrainedModel = mlContext.Model.Load(modelFile, out _);

            // Extract pretrained model parameters
            var pretrainedModelParameters =
                ((ISingleFeaturePredictionTransformer
                    <CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>)
                    pretrainedModel)
                .Model.SubModel;

            string dataFile = @"Data\preparedInput.tsv";
            DataPreparer.CreatePreparedDataFile(dataFile, onlySaveSmallSubset: false);

            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                path: dataFile,
                hasHeader: true,
                separatorChar: '\t',
                allowQuoting: true
            );

            // Load data preparation pipeline
            ITransformer dataPrepPipeline = mlContext.Model.Load(dataPreparePipelineFile, out _);

            // Prepare input data to a form consumable by a machine learning model
            var newData = dataPrepPipeline.Transform(trainingDataView);

            // Retrain model
            Console.WriteLine("Start retraining model");

            var startTime = DateTime.Now;

            var retrainedModel =
                mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression()
                    .Fit(newData, pretrainedModelParameters);

            Console.WriteLine($"Model retraining finished in {(DateTime.Now - startTime).TotalSeconds} seconds");

            return retrainedModel;
        }

        private static void EvaluateModel(MLContext mlContext, ITransformer trainedModel, IDataView testData)
        {
            Console.WriteLine();
            Console.WriteLine("-- Evaluating binary classification model performance --");
            Console.WriteLine();

            var predictedData = trainedModel.Transform(testData);
            var metrics = mlContext.BinaryClassification.Evaluate(predictedData);
            Console.WriteLine($"Accuracy: {metrics.Accuracy:0.###}");

            Console.WriteLine();
            Console.WriteLine("Confusion Matrix");
            Console.WriteLine();
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
            Console.WriteLine();
        }
    }
}