import { BotDetectorConfig } from '../types/types';

export const DEFAULT_MODEL_CONFIG: BotDetectorConfig = {
    vocabSize: 10000,
    maxSequenceLength: 100,
    embeddingDim: 128
};

export const TRAINING_DEFAULTS = {
    epochs: 10,
    batchSize: 32,
    validationSplit: 0.2
};

export const MODEL_LAYERS_CONFIG = {
    conv: {
        filters: 64,
        kernelSize: 5
    },
    lstm: {
        firstLayer: 64,
        secondLayer: 32
    },
    dense: {
        units: 64,
        dropout: 0.3
    }
};