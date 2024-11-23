import * as tf from '@tensorflow/tfjs';
import { TrainingData, TrainingOptions } from '../types/types';
import { Logger } from '../utils/logger';
import { TensorUtils } from '../utils/tensorUtils';
import { TRAINING_DEFAULTS } from '../model/modelConfig';

export class TrainingService {
    static async trainModel(
        model: tf.LayersModel,
        data: TrainingData,
        options: TrainingOptions = {}
    ): Promise<tf.History> {
        const { texts, labels } = data;
        const {
            epochs = TRAINING_DEFAULTS.epochs,
            batchSize = TRAINING_DEFAULTS.batchSize,
            validationSplit = TRAINING_DEFAULTS.validationSplit
        } = options;

        const xs = TensorUtils.prepareInputData(texts);
        const ys = TensorUtils.prepareLabels(labels);

        try {
            Logger.info('Rozpoczynam trenowanie modelu...');
            const history = await model.fit(xs, ys, {
                epochs,
                batchSize,
                validationSplit,
                callbacks: {
                    onEpochEnd: (epoch, logs) => Logger.training(epoch, logs)
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
}