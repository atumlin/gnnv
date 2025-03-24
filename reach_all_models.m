% Script for getting output set for all PF analysis models 

modelDir = 'models/';
filePath = dir(modelDir + "gcn_ieee24_*.mat");
fileName = filePath(1).name;


reach_model(fileName)