import chalk from 'chalk';

import { RussianBotDetector } from './model/RussianBotDetector';
import { trainingService } from './services/trainingService';
import { Logger } from './utils/logger';

import { ModelParams, TrainingData } from './types/types';
import { DEFAULT_MODEL_CONFIG } from './model/modelConfig';
import { TextTokenizer } from './utils/textTokenizer';
import path from 'path';

export async function runModel(params?: ModelParams): Promise<void> {
    Logger.info('STARTED', 'green');

    const detectorModel = new RussianBotDetector(DEFAULT_MODEL_CONFIG);
    const tokenizer = new TextTokenizer(DEFAULT_MODEL_CONFIG);

    if(params?.train){
        Logger.info('TRAINING', 'blue');
        
        if (params.data) {
            //Tokenize the training texts
            const tokenizedTexts = await tokenizer.batchTokenize(params.data.texts);
            const processedTrainingData: TrainingData = {
                texts: tokenizedTexts,
                labels: params.data.labels
            };
        
            await trainingService.trainModel(detectorModel.model, processedTrainingData, {
                epochs: 10,
                batchSize: 32,
                validationSplit: 0.2
            });

            const modelPath = path.resolve(process.cwd(), 'output', 'model');

            await detectorModel.saveModel(modelPath);
        }
    }
    
    if(params?.run){
        Logger.info('RUNNING', 'blue');  
    }
}

(async () => {    
    let args = process.argv.slice(2); // Remove first two elements (node and script path)
    const params: ModelParams = {};

    if(!args.length){
        args = ['run'];
    }

    if (args.includes('train')) {
        params.train = true;
    }
    if (args.includes('run')) {
        params.run = true;
    }    

    // Example data - only include if training
    if (params.train) {
        const trainingData = {
            texts: [
                "Example propaganda text 1",
                "Example propaganda text 2",
                // ... more examples
            ],
            labels: [1, 1]
        };
        
        params.data = trainingData;        
    }
    
    await runModel(params);
})();