import { promises as fs } from 'fs';
import path from 'path';
import { ModelParams, TrainingData } from './types/types';
import { RussianBotDetector } from './model/RussianBotDetector';
import { TextTokenizer } from './utils/textTokenizer';
import { DEFAULT_MODEL_CONFIG } from './model/modelConfig';
import { Logger } from './utils/logger';

async function loadTrainingData(dataPath: string): Promise<ModelParams['data']> {
    try {
        const rawData = await fs.readFile(dataPath, 'utf8');
        const data = JSON.parse(rawData);
        
        if (!data.texts || !data.labels || !Array.isArray(data.texts) || !Array.isArray(data.labels)) {
            throw new Error('Invalid training data format');
        }
        
        return data;
    } catch (error) {
        Logger.error('Error loading training data:', error as Error);
        throw error;
    }
}
async function runModel(params?: ModelParams): Promise<void> {
    Logger.info('STARTED', 'green');

    const detectorModel = new RussianBotDetector(DEFAULT_MODEL_CONFIG);
    const tokenizer = new TextTokenizer(DEFAULT_MODEL_CONFIG);

    if (params?.train) {
        Logger.info('TRAINING', 'blue');
        
        if (!params.data && process.argv[2]) {
            // Load training data from JSON file specified in CLI args
            const dataPath = process.argv[2];
            params.data = await loadTrainingData(dataPath);
        }

        if (params.data) {
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
        } else {
            throw new Error('No training data provided. Please provide a JSON file path as argument.');
        }
    }
    
    if (params?.run) {
        Logger.info('RUNNING', 'blue');  
    }
}

// Add CLI handling
if (require.main === module) {
    const args = process.argv.slice(2);
    if (args.length === 0) {
        console.log('Usage: bun run backend/index.ts <training-data.json>');
        process.exit(1);
    }
    
    runModel({ train: true }).catch(error => {
        Logger.error('Error running model:', error);
        process.exit(1);
    });
}

export { runModel };
