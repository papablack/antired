export interface BotDetectorConfig {
    vocabSize: number;
    maxSequenceLength: number;
    embeddingDim: number;
}

export interface TrainingData {
    texts: number[][];
    labels: number[];
}

export interface TrainingOptions {
    epochs?: number;
    batchSize?: number;
    validationSplit?: number;
}

export interface PredictionResult {
    probability: number;
    isBot: boolean;
    confidence: 'high' | 'medium' | 'low';
}