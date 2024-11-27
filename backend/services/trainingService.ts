import * as tf from '@tensorflow/tfjs';
import { TrainingData, TrainingOptions } from '../types/types';
import { Logger } from '../utils/logger';
import { TensorUtils } from '../utils/tensorUtils';
import { TRAINING_DEFAULTS } from '../model/modelConfig';
import { TextTokenizer } from '../utils/textTokenizer';
import { DEFAULT_MODEL_CONFIG } from '../model/modelConfig';
import { TrainingHelper } from '../utils/train';

export type TFLogType = { loss: number; acc: number };

export class TrainingServiceHolder {
    private static instance: TrainingServiceHolder;
    private tokenizer: TextTokenizer;

    private constructor() {
        this.tokenizer = new TextTokenizer(DEFAULT_MODEL_CONFIG);
    }

    public static getInstance(): TrainingServiceHolder {
        if (!TrainingServiceHolder.instance) {
            TrainingServiceHolder.instance = new TrainingServiceHolder();
        }
        return TrainingServiceHolder.instance;
    }

    async trainModel(
        model: tf.LayersModel,
        data: TrainingData,
        options: TrainingOptions = {}
    ): Promise<tf.History> {
        return TrainingHelper.trainModel(model, data, options)
    }
}

export const trainingService: TrainingServiceHolder = TrainingServiceHolder.getInstance();