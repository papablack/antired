import * as tf from '@tensorflow/tfjs';
import { TrainingData, TrainingOptions } from '../types/types';
import { Logger } from '../utils/logger';
import { TensorUtils } from '../utils/tensorUtils';
import { TRAINING_DEFAULTS } from '../model/modelConfig';

export type TFLogType = { loss: number; acc: number };

export class TrainingServiceHolder {
    private static instance: TrainingServiceHolder;

    private constructor() {
        // Private constructor to prevent direct construction calls with the `new` operator.
    }

    public static getInstance(): TrainingServiceHolder {
        if (!TrainingServiceHolder.instance) {
            TrainingServiceHolder.instance = new TrainingServiceHolder();
        }
        return TrainingServiceHolder.instance;
    }

    async trainModel(
        model: tf.LayersModel,
        data: TrainingData | { texts: string[], labels: number[] },
        options: TrainingOptions = {}
    ): Promise<tf.History> {
        // Convert string texts to number arrays if needed
        const processedData: TrainingData = {
            texts: Array.isArray(data.texts[0]) 
                ? data.texts as number[][] 
                : (data.texts as string[]).map(text => this.textToNumberArray(text)),
            labels: data.labels
        };
        const { texts, labels } = processedData;
        const {
            epochs = TRAINING_DEFAULTS.epochs,
            batchSize = TRAINING_DEFAULTS.batchSize,
            validationSplit = TRAINING_DEFAULTS.validationSplit
        } = options;

        const xs = TensorUtils.prepareInputData(texts as number[][]);
        const ys = TensorUtils.prepareLabels(labels);

        try {
            Logger.info('Rozpoczynam trenowanie modelu...');
            const history = await model.fit(xs, ys, {
                epochs,
                batchSize,
                validationSplit,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {                
                        return Logger.training(epoch, logs as TFLogType)
                    }
                }
            });
            Logger.info('Trenowanie zakończone pomyślnie!');
            return history;
        } catch (error) {
            Logger.error('Błąd podczas trenowania:', error as Error);
            throw error;
        } finally {
            TensorUtils.dispose(xs, ys);
        }
    }

    private textToNumberArray(text: string): number[] {
        // This is a placeholder implementation
        // You should implement proper text tokenization here
        // based on your vocabulary and tokenization strategy
        return text.split('').map(char => char.charCodeAt(0));
    }
}

export const trainingService: TrainingServiceHolder = TrainingServiceHolder.getInstance();
