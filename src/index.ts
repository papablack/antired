import chalk from 'chalk';

import { RussianBotDetector } from './model/RussianBotDetector';
import { TrainingService } from './services/trainingService';
import { Logger } from './utils/logger';

import {} from './types/types';
import { DEFAULT_MODEL_CONFIG } from './model/modelConfig';

export async function runModel(params?: any){
    Logger.info('STARTED', 'green');

    if(params.train){
        Logger.info('TRAINING', 'blue');
    }

    if(params.run){
        Logger.info('RUNNING', 'blue');
    }
}

(async () => {    
    let args = process.argv.slice(2); // Remove first two elements (node and script path)
    const params: any = {};

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
        params.data = {
            texts: [
                "Example propaganda text 1",
                "Example propaganda text 2",
                // ... more examples
            ],
            labels: [1, 1] // 1 for bot/propaganda, 0 for normal text
        };

        Logger.dump({ trainingData: params.data })
    }

    await runModel(params);
})();