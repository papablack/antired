import { promises as fs } from 'fs';
import path from 'path';
import { ModelParams, TrainingData } from './types/types';
import { RussianBotDetector } from './model/RussianBotDetector';
import { TextTokenizer } from './utils/textTokenizer';
import { DEFAULT_MODEL_CONFIG } from './model/modelConfig';
import { Logger } from './utils/logger';
import { trainingService } from './services/trainingService';
import { setupGPU } from './utils/gpuSetup';

async function loadTrainingData(dataPath: string): Promise<ModelParams['data']> {
    try {
        const rawData = await fs.readFile(dataPath, 'utf8');
        const data = JSON.parse(rawData);
        
        // Update validation to include languages
        if (!data.texts || !data.labels || !data.languages || 
            !Array.isArray(data.texts) || !Array.isArray(data.labels) || !Array.isArray(data.languages)) {
            throw new Error('Invalid training data format - must include texts, labels, and languages arrays');
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
        
        if (!params.data && process.argv[3]) {
            // Load training data from JSON file specified in CLI args
            const dataPath = path.resolve(__dirname, '..', 'data',process.argv[3] + '.json');
            params.data = await loadTrainingData(dataPath);
        }

        if (params.data) {
            // Tokenize with language support
            const tokenizedTexts = await tokenizer.batchTokenize(params.data.texts);
            const processedTrainingData: TrainingData = {
                texts: tokenizer.convertSequencesToStrings(tokenizedTexts),
                labels: params.data.labels,
                languages: params.data.languages // Include languages in processed data
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

    // setupGPU().then(() => {
       
    // });
    
    runModel({ train: true }).catch(error => {
        Logger.error('Error running model:', error);
        process.exit(1);
    });
}

export { runModel };
